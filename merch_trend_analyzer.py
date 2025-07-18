import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pytrends.request import TrendReq
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
import random
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")

@dataclass
class TrendScoreWeights:
    """Configuration for TrendScore calculation weights"""
    bsr_weight: float = 0.25
    rating_weight: float = 0.20
    review_weight: float = 0.20
    price_trend_weight: float = 0.15
    google_trends_weight: float = 0.20

@dataclass
class ProductMetrics:
    """Data class for product metrics"""
    asin: str
    title: str
    bsr: int
    price: float
    rating: float
    review_count: int
    normalized_bsr: float
    normalized_rating: float
    normalized_reviews: float
    price_trend: float
    google_trend_momentum: float
    trend_score: float
    keywords: List[str]
    niche: str

class AmazonAnalyzer:
    """Core Amazon Merch analyzer with TrendScore calculation and Google Trends integration"""
    
    def __init__(self, weights: TrendScoreWeights = None):
        self.weights = weights or TrendScoreWeights()
        self.df = None
        self.processed_data = []
        self.top_products = []
        self.niches = {}
        self.pytrends = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.trends_cache = {}  # Cache for Google Trends data
        self.request_count = 0
        self.last_request_time = time.time()
        self._initialize_trends()
        
    def _initialize_trends(self):
        """Initialize Google Trends connection with robust error handling"""
        try:
            self.pytrends = TrendReq(
                hl='en-US', 
                tz=360, 
                timeout=(10, 25), 
                retries=2, 
                backoff_factor=0.1,
                requests_args={'verify': False}  # Handle SSL issues
            )
            logger.info("Google Trends (pytrends) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Trends: {e}")
            self.pytrends = None
    
    def load_csv(self, file_path: str) -> bool:
        """Load and validate Amazon Merch CSV data"""
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Log available columns
            logger.info(f"Available columns: {list(self.df.columns)}")
            
            # Try to map common column variations
            column_mapping = {
                'title': 'Title',
                'product_title': 'Title',
                'name': 'Title',
                'asin': 'ASIN',
                'product_id': 'ASIN',
                'bsr': 'BSR',
                'best_sellers_rank': 'BSR',
                'rank': 'BSR',
                'price': 'Price',
                'product_price': 'Price',
                'rating': 'Rating',
                'average_rating': 'Rating',
                'review_count': 'Review Count',
                'reviews': 'Review Count',
                'number_of_reviews': 'Review Count'
            }
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in self.df.columns and new_col not in self.df.columns:
                    self.df.rename(columns={old_col: new_col}, inplace=True)
            
            # Validate required columns
            required_columns = ['Title', 'ASIN', 'BSR', 'Price', 'Rating', 'Review Count']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.info(f"Available columns: {list(self.df.columns)}")
                return False
                
            # Clean and standardize data
            self._clean_data()
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return False
    
    def _clean_data(self):
        """Clean and normalize the dataset"""
        initial_count = len(self.df)
        
        # Remove rows with missing critical data
        self.df = self.df.dropna(subset=['Title', 'ASIN'])
        
        # Fill missing values with defaults
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce').fillna(0)
        self.df['Rating'] = pd.to_numeric(self.df['Rating'], errors='coerce').fillna(0)
        self.df['Review Count'] = pd.to_numeric(self.df['Review Count'], errors='coerce').fillna(0)
        self.df['BSR'] = pd.to_numeric(self.df['BSR'], errors='coerce').fillna(999999)
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['ASIN'])
        
        # Clean titles
        self.df['Title'] = self.df['Title'].astype(str).str.strip()
        
        # Remove products with invalid data
        self.df = self.df[
            (self.df['Title'].str.len() > 0) & 
            (self.df['BSR'] > 0) & 
            (self.df['BSR'] < 10000000)  # Reasonable BSR limit
        ]
        
        logger.info(f"Cleaned dataset: {len(self.df)} products remaining (removed {initial_count - len(self.df)} invalid records)")
    
    def extract_keywords(self, title: str, max_keywords: int = 10) -> List[str]:
        """Extract meaningful keywords from product titles using NLP"""
        try:
            # Convert to lowercase and remove special characters
            title_clean = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower())
            
            # Tokenize
            tokens = word_tokenize(title_clean)
            
            # Remove stopwords and short words
            keywords = [
                self.lemmatizer.lemmatize(word) 
                for word in tokens 
                if word not in self.stop_words 
                and len(word) > 2
                and not word.isdigit()
                and word.isalpha()  # Only alphabetic words
            ]
            
            # Remove duplicates while preserving order
            unique_keywords = []
            seen = set()
            for keyword in keywords:
                if keyword not in seen:
                    unique_keywords.append(keyword)
                    seen.add(keyword)
            
            return unique_keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords from '{title}': {e}")
            return []
    
    def _throttle_requests(self):
        """Implement smart request throttling for Google Trends"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Adaptive delay based on request count
        if self.request_count > 10:
            delay = random.uniform(3, 6)
        elif self.request_count > 5:
            delay = random.uniform(2, 4)
        else:
            delay = random.uniform(1, 2)
        
        # Ensure minimum delay
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            logger.debug(f"Throttling request: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_google_trends_data(self, keywords: List[str], timeframe: str = "today 3-m") -> float:
        """Get Google Trends momentum with proper throttling and caching"""
        if not self.pytrends or not keywords:
            return 0.0
        
        # Create cache key
        cache_key = f"{'-'.join(sorted(keywords[:3]))}-{timeframe}"
        
        # Check cache first
        if cache_key in self.trends_cache:
            logger.debug(f"Using cached trends data for {keywords[:3]}")
            return self.trends_cache[cache_key]
        
        try:
            # Throttle requests
            self._throttle_requests()
            
            # Clean and limit keywords
            clean_keywords = []
            for keyword in keywords[:3]:  # Limit to 3 keywords to avoid API limits
                if len(keyword) > 2 and keyword.isalpha():
                    clean_keywords.append(keyword)
            
            if not clean_keywords:
                return 0.0
            
            logger.debug(f"Fetching Google Trends for: {clean_keywords}")
            
            # Build payload with error handling
            self.pytrends.build_payload(clean_keywords, timeframe=timeframe, geo='US')
            
            # Get interest over time
            data = self.pytrends.interest_over_time()
            
            if data.empty or len(data) < 2:
                momentum = 0.0
            else:
                # Calculate momentum (trend slope)
                values = data.iloc[:, :-1].sum(axis=1) if 'isPartial' in data.columns else data.sum(axis=1)
                
                if len(values) < 2:
                    momentum = 0.0
                else:
                    # Calculate linear trend
                    x = np.arange(len(values))
                    try:
                        slope = np.polyfit(x, values, 1)[0]
                        # Normalize slope to 0-1 range
                        momentum = max(0, min(1, (slope + 10) / 20))
                    except:
                        momentum = 0.0
            
            # Cache the result
            self.trends_cache[cache_key] = momentum
            
            logger.debug(f"Google Trends momentum for {clean_keywords}: {momentum:.4f}")
            return momentum
            
        except Exception as e:
            logger.warning(f"Google Trends error for {keywords}: {e}")
            # Return neutral momentum on error
            momentum = 0.5
            self.trends_cache[cache_key] = momentum
            return momentum
    
    def get_trend_score_with_fallback(self, keywords: List[str]) -> float:
        """Get trend score with fallback to keyword-based scoring"""
        
        # Try Google Trends first
        google_score = self.get_google_trends_data(keywords)
        
        if google_score > 0:
            return google_score
        
        # Fallback: keyword-based trend estimation
        trending_keywords = {
            'christmas': 0.9, 'halloween': 0.8, 'thanksgiving': 0.7,
            'funny': 0.7, 'humor': 0.7, 'joke': 0.6,
            'dog': 0.6, 'cat': 0.6, 'pet': 0.6,
            'mom': 0.5, 'dad': 0.5, 'family': 0.5,
            'coffee': 0.6, 'wine': 0.6, 'beer': 0.6,
            'teacher': 0.5, 'nurse': 0.5, 'doctor': 0.5,
            'vintage': 0.4, 'retro': 0.4, 'classic': 0.4,
            'love': 0.5, 'heart': 0.5, 'peace': 0.4
        }
        
        keyword_scores = []
        for keyword in keywords:
            score = trending_keywords.get(keyword.lower(), 0.3)
            keyword_scores.append(score)
        
        fallback_score = np.mean(keyword_scores) if keyword_scores else 0.3
        logger.debug(f"Using fallback trend score: {fallback_score:.4f} for {keywords}")
        return fallback_score
    
    def calculate_price_trend(self, current_price: float, historical_prices: List[float] = None) -> float:
        """Calculate price trend (enhanced with market positioning)"""
        # If historical data available, calculate actual trend
        if historical_prices and len(historical_prices) >= 2:
            recent_avg = np.mean(historical_prices[-3:])
            older_avg = np.mean(historical_prices[:-3])
            
            if older_avg > 0:
                trend = (recent_avg - older_avg) / older_avg
                return max(0, min(1, (trend + 0.5) / 1.0))
        
        # Enhanced price trend simulation based on market positioning
        if current_price < 10:
            return 0.8  # Very low prices - high demand potential
        elif current_price < 15:
            return 0.7  # Low prices - good demand
        elif current_price < 20:
            return 0.6  # Mid-low prices - decent demand
        elif current_price < 25:
            return 0.5  # Mid prices - stable
        elif current_price < 30:
            return 0.4  # Mid-high prices - declining demand
        else:
            return 0.3  # High prices - lower demand
    
    def calculate_trend_score(self, product_data: Dict) -> float:
        """Calculate TrendScore using weighted formula with enhanced normalization"""
        try:
            # Enhanced BSR normalization with logarithmic scale
            max_bsr = max(self.df['BSR'].max(), 1000000)
            min_bsr = max(self.df['BSR'].min(), 1)
            
            # Use logarithmic scale for BSR (lower BSR is better)
            log_bsr = np.log(product_data['BSR'])
            log_max_bsr = np.log(max_bsr)
            log_min_bsr = np.log(min_bsr)
            
            normalized_bsr = 1 - ((log_bsr - log_min_bsr) / (log_max_bsr - log_min_bsr))
            normalized_bsr = max(0, min(1, normalized_bsr))
            
            # Enhanced rating normalization
            normalized_rating = max(0, min(1, product_data['Rating'] / 5.0))
            
            # Enhanced review count normalization with logarithmic scale
            max_reviews = max(self.df['Review Count'].max(), 100)
            normalized_reviews = np.log(product_data['Review Count'] + 1) / np.log(max_reviews + 1)
            
            # Calculate TrendScore with validation
            trend_score = (
                self.weights.bsr_weight * normalized_bsr +
                self.weights.rating_weight * normalized_rating +
                self.weights.review_weight * normalized_reviews +
                self.weights.price_trend_weight * product_data['price_trend'] +
                self.weights.google_trends_weight * product_data['google_momentum']
            )
            
            # Ensure score is in valid range
            trend_score = max(0.0, min(1.0, trend_score))
            
            logger.debug(f"TrendScore calculation - BSR: {normalized_bsr:.3f}, Rating: {normalized_rating:.3f}, "
                        f"Reviews: {normalized_reviews:.3f}, Price: {product_data['price_trend']:.3f}, "
                        f"Google: {product_data['google_momentum']:.3f}, Final: {trend_score:.3f}")
            
            return trend_score
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0
    
    def process_products(self, batch_size: int = 10, use_google_trends: bool = True, 
                        delay_between_batches: float = 30.0) -> List[ProductMetrics]:
        """Process all products with enhanced batch processing and error handling"""
        if self.df is None:
            logger.error("No data loaded")
            return []
        
        processed_products = []
        total_products = len(self.df)
        
        logger.info(f"Processing {total_products} products in batches of {batch_size}")
        
        for batch_start in range(0, total_products, batch_size):
            batch_end = min(batch_start + batch_size, total_products)
            batch_df = self.df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_products-1)//batch_size + 1} "
                       f"(products {batch_start+1}-{batch_end})")
            
            for index, row in batch_df.iterrows():
                try:
                    # Extract keywords
                    keywords = self.extract_keywords(row['Title'])
                    
                    if not keywords:
                        logger.warning(f"No keywords extracted for product: {row['Title']}")
                        continue
                    
                    # Get Google Trends momentum
                    if use_google_trends:
                        google_momentum = self.get_trend_score_with_fallback(keywords)
                    else:
                        google_momentum = self.get_trend_score_with_fallback(keywords)  # Uses fallback
                    
                    # Calculate price trend
                    price_trend = self.calculate_price_trend(row['Price'])
                    
                    # Prepare product data for scoring
                    product_data = {
                        'BSR': row['BSR'],
                        'Rating': row['Rating'],
                        'Review Count': row['Review Count'],
                        'price_trend': price_trend,
                        'google_momentum': google_momentum
                    }
                    
                    # Calculate TrendScore
                    trend_score = self.calculate_trend_score(product_data)
                    
                    # Determine niche
                    niche = self._determine_niche(keywords)
                    
                    # Create ProductMetrics object
                    product_metrics = ProductMetrics(
                        asin=row['ASIN'],
                        title=row['Title'],
                        bsr=int(row['BSR']),
                        price=float(row['Price']),
                        rating=float(row['Rating']),
                        review_count=int(row['Review Count']),
                        normalized_bsr=1 - (row['BSR'] / self.df['BSR'].max()),
                        normalized_rating=row['Rating'] / 5.0,
                        normalized_reviews=np.log(row['Review Count'] + 1) / np.log(self.df['Review Count'].max() + 1),
                        price_trend=price_trend,
                        google_trend_momentum=google_momentum,
                        trend_score=trend_score,
                        keywords=keywords,
                        niche=niche
                    )
                    
                    processed_products.append(product_metrics)
                    
                except Exception as e:
                    logger.error(f"Error processing product {row.get('ASIN', 'unknown')}: {e}")
                    continue
            
            # Delay between batches to avoid rate limiting
            if batch_end < total_products and use_google_trends:
                logger.info(f"Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
        
        self.processed_data = processed_products
        logger.info(f"Successfully processed {len(processed_products)} products")
        return processed_products
    
    def _determine_niche(self, keywords: List[str]) -> str:
        """Determine product niche based on keywords with enhanced categorization"""
        niche_keywords = {
            'funny': ['funny', 'humor', 'joke', 'comedy', 'laugh', 'hilarious', 'meme', 'sarcastic'],
            'political': ['trump', 'biden', 'politics', 'election', 'vote', 'political', 'america', 'patriot'],
            'pet': ['dog', 'cat', 'pet', 'puppy', 'kitten', 'animal', 'paw', 'breed'],
            'sports': ['football', 'basketball', 'soccer', 'baseball', 'sports', 'team', 'player', 'game'],
            'family': ['mom', 'dad', 'family', 'parent', 'kids', 'children', 'mother', 'father'],
            'motivational': ['motivational', 'inspiration', 'success', 'dream', 'achieve', 'believe', 'hustle'],
            'holiday': ['christmas', 'halloween', 'thanksgiving', 'valentine', 'holiday', 'easter', 'birthday'],
            'occupation': ['teacher', 'nurse', 'doctor', 'engineer', 'job', 'work', 'professional', 'career'],
            'food_drink': ['coffee', 'wine', 'beer', 'food', 'drink', 'cooking', 'chef', 'kitchen'],
            'music': ['music', 'guitar', 'piano', 'band', 'rock', 'jazz', 'classical', 'musician'],
            'travel': ['travel', 'vacation', 'adventure', 'explore', 'world', 'journey', 'wanderlust'],
            'fitness': ['fitness', 'gym', 'workout', 'strong', 'muscle', 'health', 'exercise', 'yoga'],
            'vintage': ['vintage', 'retro', 'classic', 'old', 'antique', 'throwback', 'nostalgic']
        }
        
        keyword_str = ' '.join(keywords).lower()
        
        # Score each niche
        niche_scores = {}
        for niche, niche_words in niche_keywords.items():
            score = sum(1 for word in niche_words if word in keyword_str)
            if score > 0:
                niche_scores[niche] = score
        
        # Return the niche with highest score
        if niche_scores:
            return max(niche_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def get_top_products(self, top_n: int = 10, min_score: float = 0.0) -> List[ProductMetrics]:
        """Get top N trending products with optional minimum score filter"""
        if not self.processed_data:
            logger.error("No processed data available")
            return []
        
        # Filter by minimum score if specified
        if min_score > 0:
            filtered_products = [p for p in self.processed_data if p.trend_score >= min_score]
            logger.info(f"Filtered {len(filtered_products)} products with score >= {min_score}")
        else:
            filtered_products = self.processed_data
        
        # Sort by TrendScore in descending order
        sorted_products = sorted(filtered_products, key=lambda x: x.trend_score, reverse=True)
        self.top_products = sorted_products[:top_n]
        
        logger.info(f"Top {len(self.top_products)} products identified")
        return self.top_products
    
    def analyze_niches(self) -> Dict[str, Dict]:
        """Analyze niche performance and trends with enhanced metrics"""
        if not self.processed_data:
            return {}
        
        niche_analysis = {}
        
        for product in self.processed_data:
            niche = product.niche
            if niche not in niche_analysis:
                niche_analysis[niche] = {
                    'count': 0,
                    'avg_score': 0,
                    'avg_bsr': 0,
                    'avg_rating': 0,
                    'total_reviews': 0,
                    'avg_price': 0,
                    'avg_google_momentum': 0,
                    'top_keywords': [],
                    'score_distribution': [],
                    'top_products': []
                }
            
            niche_data = niche_analysis[niche]
            niche_data['count'] += 1
            niche_data['avg_score'] += product.trend_score
            niche_data['avg_bsr'] += product.bsr
            niche_data['avg_rating'] += product.rating
            niche_data['total_reviews'] += product.review_count
            niche_data['avg_price'] += product.price
            niche_data['avg_google_momentum'] += product.google_trend_momentum
            niche_data['top_keywords'].extend(product.keywords)
            niche_data['score_distribution'].append(product.trend_score)
            niche_data['top_products'].append(product)
        
        # Calculate averages and statistics
        for niche, data in niche_analysis.items():
            count = data['count']
            data['avg_score'] /= count
            data['avg_bsr'] /= count
            data['avg_rating'] /= count
            data['avg_price'] /= count
            data['avg_google_momentum'] /= count
            
            # Score statistics
            scores = data['score_distribution']
            data['score_std'] = np.std(scores)
            data['score_median'] = np.median(scores)
            data['score_min'] = min(scores)
            data['score_max'] = max(scores)
            
            # Get top keywords for niche
            keyword_freq = {}
            for keyword in data['top_keywords']:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            data['top_keywords'] = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Keep only top 5 products for this niche
            data['top_products'] = sorted(data['top_products'], key=lambda x: x.trend_score, reverse=True)[:5]
        
        self.niches = niche_analysis
        return niche_analysis
    
    def cluster_products(self, n_clusters: int = 5) -> Dict[int, List[ProductMetrics]]:
        """Cluster products using KMeans with enhanced features"""
        if not self.processed_data:
            return {}
        
        # Prepare features for clustering
        features = []
        for product in self.processed_data:
            features.append([
                product.normalized_bsr,
                product.normalized_rating,
                product.normalized_reviews,
                product.price_trend,
                product.google_trend_momentum,
                product.price / 50.0,  # Normalized price
                len(product.keywords) / 10.0,  # Keyword count
            ])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Group products by cluster
        clustered_products = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_products:
                clustered_products[cluster_id] = []
            clustered_products[cluster_id].append(self.processed_data[i])
        
        # Add cluster statistics
        for cluster_id, products in clustered_products.items():
            avg_score = np.mean([p.trend_score for p in products])
            avg_bsr = np.mean([p.bsr for p in products])
            avg_price = np.mean([p.price for p in products])
            
            logger.info(f"Cluster {cluster_id}: {len(products)} products, "
                       f"Avg Score: {avg_score:.3f}, Avg BSR: {avg_bsr:.0f}, Avg Price: ${avg_price:.2f}")
        
        return clustered_products
    
    def export_results(self, file_path: str, format: str = 'csv', include_details: bool = True) -> bool:
        """Export results with enhanced data and multiple formats"""
        if not self.top_products:
            logger.error("No top products to export")
            return False
        
        try:
            if format.lower() == 'csv':
                export_data = []
                for rank, product in enumerate(self.top_products, 1):
                    base_data = {
                        'Rank': rank,
                        'ASIN': product.asin,
                        'Title': product.title,
                        'BSR': product.bsr,
                        'Price': product.price,
                        'Rating': product.rating,
                        'Review Count': product.review_count,
                        'TrendScore': round(product.trend_score, 4),
                        'Niche': product.niche,
                        'Keywords': ', '.join(product.keywords[:5])
                    }
                    
                    if include_details:
                        base_data.update({
                            'Google Momentum': round(product.google_trend_momentum, 4),
                            'Price Trend': round(product.price_trend, 4),
                            'Normalized BSR': round(product.normalized_bsr, 4),
                            'Normalized Rating': round(product.normalized_rating, 4),
                            'Normalized Reviews': round(product.normalized_reviews, 4),
                            'All Keywords': ', '.join(product.keywords)
                        })
                    
                    export_data.append(base_data)
                
                df_export = pd.DataFrame(export_data)
                df_export.to_csv(file_path, index=False)
                logger.info(f"Results exported to {file_path}")
                return True
            
            elif format.lower() == 'json':
                export_data = []
                for rank, product in enumerate(self.top_products, 1):
                    product_data = {
                        'rank': rank,
                        'asin': product.asin,
                        'title': product.title,
                        'bsr': product.bsr,
                        'price': product.price,
                        'rating': product.rating,
                        'review_count': product.review_count,
                        'trend_score': round(product.trend_score, 4),
                        'niche': product.niche,
                        'keywords': product.keywords[:5]
                    }
                    
                    if include_details:
                        product_data.update({
                            'google_momentum': round(product.google_trend_momentum, 4),
                            'price_trend': round(product.price_trend, 4),
                            'normalized_bsr': round(product.normalized_bsr, 4),
                            'normalized_rating': round(product.normalized_rating, 4),
                            'normalized_reviews': round(product.normalized_reviews, 4),
                            'all_keywords': product.keywords
                        })
                    
                    export_data.append(product_data)
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Results exported to {file_path}")
                return True
            
            else:
                logger.error(f"Unsupported format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def generate_report(self, file_path: str = None) -> str:
        """Generate comprehensive analysis report"""
        if not self.processed_data:
            return "No data processed yet"
        
        report = []
        report.append("=" * 60)
        report.append("AMAZON MERCH TREND ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Products Analyzed: {len(self.processed_data)}")
        report.append("")
        
        # Overall statistics
        scores = [p.trend_score for p in self.processed_data]
        report.append("OVERALL STATISTICS")
        report.append("-" * 30)
        report.append(f"Average TrendScore: {np.mean(scores):.4f}")
        report.append(f"Median TrendScore: {np.median(scores):.4f}")
        report.append(f"Standard Deviation: {np.std(scores):.4f}")
        report.append(f"Min Score: {min(scores):.4f}")
        report.append(f"Max Score: {max(scores):.4f}")
        report.append("")
        
        # Top products
        if self.top_products:
            report.append("TOP TRENDING PRODUCTS")
            report.append("-" * 30)
            for i, product in enumerate(self.top_products[:5], 1):
                report.append(f"{i}. {product.title[:60]}...")
                report.append(f"   ASIN: {product.asin}")
                report.append(f"   TrendScore: {product.trend_score:.4f}")
                report.append(f"   BSR: {product.bsr:,}")
                report.append(f"   Price: ${product.price:.2f}")
                report.append(f"   Rating: {product.rating}/5.0 ({product.review_count} reviews)")
                report.append(f"   Niche: {product.niche}")
                report.append(f"   Keywords: {', '.join(product.keywords[:5])}")
                report.append("")
        
        # Niche analysis
        if self.niches:
            report.append("NICHE ANALYSIS")
            report.append("-" * 30)
            sorted_niches = sorted(self.niches.items(), key=lambda x: x[1]['avg_score'], reverse=True)
            
            for niche, data in sorted_niches[:10]:
                report.append(f"{niche.upper()}:")
                report.append(f"  Products: {data['count']}")
                report.append(f"  Avg Score: {data['avg_score']:.4f}")
                report.append(f"  Avg BSR: {data['avg_bsr']:,.0f}")
                report.append(f"  Avg Rating: {data['avg_rating']:.2f}")
                report.append(f"  Avg Price: ${data['avg_price']:.2f}")
                report.append(f"  Top Keywords: {', '.join([k[0] for k in data['top_keywords'][:5]])}")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        
        if self.niches:
            top_niche = max(self.niches.items(), key=lambda x: x[1]['avg_score'])
            report.append(f"1. Focus on '{top_niche[0]}' niche (highest average score: {top_niche[1]['avg_score']:.4f})")
            
            profitable_niches = [n for n, d in self.niches.items() if d['avg_score'] > 0.6 and d['count'] >= 5]
            if profitable_niches:
                report.append(f"2. Consider these profitable niches: {', '.join(profitable_niches)}")
        
        high_momentum_products = [p for p in self.processed_data if p.google_trend_momentum > 0.7]
        if high_momentum_products:
            report.append(f"3. {len(high_momentum_products)} products show high Google Trends momentum")
        
        report.append("")
        report.append("WEIGHT CONFIGURATION USED:")
        report.append(f"  BSR Weight: {self.weights.bsr_weight}")
        report.append(f"  Rating Weight: {self.weights.rating_weight}")
        report.append(f"  Review Weight: {self.weights.review_weight}")
        report.append(f"  Price Trend Weight: {self.weights.price_trend_weight}")
        report.append(f"  Google Trends Weight: {self.weights.google_trends_weight}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {file_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report_text
    
    def get_keyword_trends(self, top_n: int = 20) -> Dict[str, Dict]:
        """Analyze keyword trends across all products"""
        if not self.processed_data:
            return {}
        
        keyword_data = {}
        
        for product in self.processed_data:
            for keyword in product.keywords:
                if keyword not in keyword_data:
                    keyword_data[keyword] = {
                        'count': 0,
                        'total_score': 0,
                        'avg_score': 0,
                        'avg_bsr': 0,
                        'avg_price': 0,
                        'products': []
                    }
                
                kw_data = keyword_data[keyword]
                kw_data['count'] += 1
                kw_data['total_score'] += product.trend_score
                kw_data['avg_bsr'] += product.bsr
                kw_data['avg_price'] += product.price
                kw_data['products'].append(product)
        
        # Calculate averages
        for keyword, data in keyword_data.items():
            count = data['count']
            data['avg_score'] = data['total_score'] / count
            data['avg_bsr'] /= count
            data['avg_price'] /= count
            
            # Keep only top products for this keyword
            data['products'] = sorted(data['products'], key=lambda x: x.trend_score, reverse=True)[:3]
        
        # Filter and sort by relevance (combination of count and average score)
        filtered_keywords = {
            k: v for k, v in keyword_data.items() 
            if v['count'] >= 2  # At least 2 products
        }
        
        sorted_keywords = sorted(
            filtered_keywords.items(), 
            key=lambda x: x[1]['avg_score'] * np.log(x[1]['count'] + 1), 
            reverse=True
        )
        
        return dict(sorted_keywords[:top_n])
    
    def update_weights(self, new_weights: TrendScoreWeights):
        """Update TrendScore weights and recalculate scores"""
        self.weights = new_weights
        
        if self.processed_data:
            logger.info("Recalculating TrendScores with new weights...")
            
            for product in self.processed_data:
                product_data = {
                    'BSR': product.bsr,
                    'Rating': product.rating,
                    'Review Count': product.review_count,
                    'price_trend': product.price_trend,
                    'google_momentum': product.google_trend_momentum
                }
                
                product.trend_score = self.calculate_trend_score(product_data)
            
            logger.info("TrendScores updated successfully")
    
    def save_state(self, file_path: str):
        """Save analyzer state to file"""
        try:
            state = {
                'weights': {
                    'bsr_weight': self.weights.bsr_weight,
                    'rating_weight': self.weights.rating_weight,
                    'review_weight': self.weights.review_weight,
                    'price_trend_weight': self.weights.price_trend_weight,
                    'google_trends_weight': self.weights.google_trends_weight
                },
                'processed_count': len(self.processed_data),
                'trends_cache': self.trends_cache,
                'niches': self.niches
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"State saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self, file_path: str):
        """Load analyzer state from file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restore weights
            weights_data = state.get('weights', {})
            self.weights = TrendScoreWeights(
                bsr_weight=weights_data.get('bsr_weight', 0.25),
                rating_weight=weights_data.get('rating_weight', 0.20),
                review_weight=weights_data.get('review_weight', 0.20),
                price_trend_weight=weights_data.get('price_trend_weight', 0.15),
                google_trends_weight=weights_data.get('google_trends_weight', 0.20)
            )
            
            # Restore cache and other data
            self.trends_cache = state.get('trends_cache', {})
            self.niches = state.get('niches', {})
            
            logger.info(f"State loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AmazonAnalyzer()
    
    # Example of loading data and processing
    print("Amazon Merch Trend Analyzer")
    print("=" * 40)
    
    # Load CSV data
    csv_file = "amazon_dataset.csv"  # Replace with your CSV file path
    if analyzer.load_csv(csv_file):
        print(f"✓ Data loaded successfully")
        
        # Process products (smaller batch for testing)
        print("Processing products...")
        analyzer.process_products(batch_size=5, delay_between_batches=10.0)
        
        # Get top products
        top_products = analyzer.get_top_products(top_n=10)
        print(f"✓ Found {len(top_products)} top products")
        
        # Analyze niches
        niches = analyzer.analyze_niches()
        print(f"✓ Analyzed {len(niches)} niches")
        
        # Export results
        analyzer.export_results("top_products.csv")
        analyzer.export_results("top_products.json", format='json')
        
        # Generate report
        report = analyzer.generate_report("analysis_report.txt")
        print("\n" + report[:500] + "...")
        
        # Save state
        analyzer.save_state("analyzer_state.json")
        
        print("\n✓ Analysis complete!")
    else:
        print("✗ Failed to load data")
        print("Please ensure your CSV file has the required columns:")
        print("Title, ASIN, BSR, Price, Rating, Review Count")