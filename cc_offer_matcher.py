import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from datetime import datetime, date
import re
import math
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class TransactionOfferMatcher:
    """
    A comprehensive system for matching credit card transactions with offers
    using both MID-based and MID-less approaches with fuzzy matching and proximity.
    """
    
    def __init__(self, proximity_radius_miles: float = 25.0):
        """
        Initialize the matcher with configurable proximity radius.
        
        Args:
            proximity_radius_miles: Maximum distance in miles for ZIP proximity matching
        """
        self.proximity_radius_miles = proximity_radius_miles
        self.zip_coordinates = {}  # Cache for ZIP to lat/long mapping
        
    def clean_merchant_name(self, name: str) -> str:
        """
        Clean and normalize merchant names for better matching.
        
        Args:
            name: Raw merchant name
            
        Returns:
            Cleaned merchant name
        """
        if pd.isna(name) or name is None:
            return ""
        
        # Convert to string and uppercase
        name = str(name).upper().strip()
        
        # Remove common prefixes/suffixes that might interfere with matching
        name = re.sub(r'\b(INC|LLC|LTD|CORP|CO|COMPANY)\b\.?', '', name)
        name = re.sub(r'\b(THE|A|AN)\b', '', name)
        
        # Remove special characters and extra spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def clean_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize location data (ZIP codes, cities, states).
        
        Args:
            df: DataFrame with location columns
            
        Returns:
            DataFrame with cleaned location data
        """
        df = df.copy()
        
        # Clean ZIP codes - keep only first 5 digits
        if 'zip_code' in df.columns:
            df['zip_code'] = df['zip_code'].astype(str).str.extract(r'(\d{5})')[0]
        
        # Clean city names
        if 'city' in df.columns:
            df['city'] = df['city'].astype(str).str.upper().str.strip()
            df['city'] = df['city'].replace('NAN', np.nan)
        
        # Clean state codes
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.upper().str.strip()
            df['state'] = df['state'].replace('NAN', np.nan)
            # Ensure 2-character state codes
            df.loc[df['state'].str.len() > 2, 'state'] = np.nan
        
        return df
    
    def parse_date(self, date_value: Union[str, datetime, date, None]) -> Optional[datetime]:
        """
        Parse various date formats into datetime objects.
        
        Args:
            date_value: Date in various formats
            
        Returns:
            Parsed datetime or None if invalid
        """
        if pd.isna(date_value) or date_value is None:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, date):
            return datetime.combine(date_value, datetime.min.time())
        
        # Try to parse string dates
        date_str = str(date_value).strip()
        
        # Common date formats
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def compute_valid_end_date(self, row: pd.Series) -> datetime:
        """
        Compute valid end date for offers based on business rules.
        
        Args:
            row: Offer row with expiration_date and deleted_at columns
            
        Returns:
            Valid end date
        """
        expiration_date = self.parse_date(row.get('expiration_date'))
        deleted_at = self.parse_date(row.get('deleted_at'))
        
        # Business rules for valid_end_date
        if expiration_date and pd.isna(row.get('deleted_at')):
            return expiration_date
        elif not expiration_date and deleted_at:
            return deleted_at
        elif expiration_date and deleted_at:
            # Use the earlier of the two dates
            return min(expiration_date, deleted_at)
        else:
            # Use max possible date
            return datetime(2099, 12, 31)
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two points on Earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            Distance in miles
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in miles
        r = 3956
        return c * r
    
    def get_zip_coordinates(self, zip_code: str) -> Optional[Tuple[float, float]]:
        """
        Get latitude and longitude for a ZIP code.
        In a real implementation, this would query a ZIP code database.
        
        Args:
            zip_code: 5-digit ZIP code
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        # This is a mock implementation - in reality you'd use a ZIP code database
        # For demo purposes, using some sample coordinates
        sample_zips = {
            '10001': (40.7505, -73.9934),  # NYC
            '90210': (34.0901, -118.4065), # Beverly Hills
            '60601': (41.8781, -87.6298),  # Chicago
            '30301': (33.7490, -84.3880),  # Atlanta
            '94102': (37.7749, -122.4194), # San Francisco
        }
        
        if zip_code in self.zip_coordinates:
            return self.zip_coordinates[zip_code]
        
        # In reality, query your ZIP database here
        coords = sample_zips.get(zip_code)
        if coords:
            self.zip_coordinates[zip_code] = coords
        
        return coords
    
    def is_zip_within_radius(self, zip1: str, zip2: str) -> bool:
        """
        Check if two ZIP codes are within the configured radius.
        
        Args:
            zip1, zip2: ZIP codes to compare
            
        Returns:
            True if within radius, False otherwise
        """
        if not zip1 or not zip2 or pd.isna(zip1) or pd.isna(zip2):
            return False
        
        coords1 = self.get_zip_coordinates(str(zip1))
        coords2 = self.get_zip_coordinates(str(zip2))
        
        if not coords1 or not coords2:
            return False
        
        distance = self.haversine_distance(
            coords1[0], coords1[1], coords2[0], coords2[1]
        )
        
        return distance <= self.proximity_radius_miles
    
    def location_matches(self, tx_row: pd.Series, offer_row: pd.Series) -> bool:
        """
        Check if transaction and offer locations match using various criteria.
        
        Args:
            tx_row: Transaction row
            offer_row: Offer row
            
        Returns:
            True if locations match, False otherwise
        """
        # Exact ZIP match
        if (not pd.isna(tx_row.get('zip_code')) and 
            not pd.isna(offer_row.get('zip_code')) and
            str(tx_row['zip_code']) == str(offer_row['zip_code'])):
            return True
        
        # ZIP within radius match
        if (not pd.isna(tx_row.get('zip_code')) and 
            not pd.isna(offer_row.get('zip_code'))):
            if self.is_zip_within_radius(tx_row['zip_code'], offer_row['zip_code']):
                return True
        
        # City + State match
        if (not pd.isna(tx_row.get('city')) and not pd.isna(tx_row.get('state')) and
            not pd.isna(offer_row.get('city')) and not pd.isna(offer_row.get('state'))):
            return (str(tx_row['city']).upper() == str(offer_row['city']).upper() and
                    str(tx_row['state']).upper() == str(offer_row['state']).upper())
        
        return False
    
    def date_in_range(self, tx_date: datetime, offer_start: datetime, offer_end: datetime) -> bool:
        """
        Check if transaction date falls within offer date range.
        
        Args:
            tx_date: Transaction date
            offer_start: Offer start date
            offer_end: Offer end date
            
        Returns:
            True if date is in range, False otherwise
        """
        if not tx_date or not offer_start:
            return False
        
        if not offer_end:
            offer_end = datetime(2099, 12, 31)
        
        return offer_start <= tx_date <= offer_end
    
    def calculate_reward_priority(self, reward_type: str, reward_value: float) -> Tuple[int, float]:
        """
        Calculate priority for reward selection (higher is better).
        
        Args:
            reward_type: Type of reward ('percentage' or 'fixed')
            reward_value: Reward value
            
        Returns:
            Tuple of (type_priority, value) for sorting
        """
        if reward_type == 'percentage':
            return (2, reward_value)  # Percentage rewards have higher priority
        else:  # fixed
            return (1, reward_value)  # Fixed rewards have lower priority
    
    def select_best_offer(self, matching_offers: List[Dict]) -> Optional[Dict]:
        """
        Select the best offer from a list of matching offers.
        
        Args:
            matching_offers: List of matching offer dictionaries
            
        Returns:
            Best offer dictionary or None if list is empty
        """
        if not matching_offers:
            return None
        
        # Sort offers by reward priority (descending)
        def sort_key(offer):
            priority, value = self.calculate_reward_priority(
                offer['reward_type'], offer['reward_value']
            )
            return (-priority, -value)  # Negative for descending sort
        
        return sorted(matching_offers, key=sort_key)[0]
    
    def clean_and_merge_offers(self, mid_offers_df: pd.DataFrame, 
                              midless_offers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean, merge, and deduplicate offer datasets.
        
        Args:
            mid_offers_df: MID-based offers
            midless_offers_df: MID-less offers
            
        Returns:
            Cleaned and merged offers DataFrame
        """
        # Clean merchant names
        mid_offers_df = mid_offers_df.copy()
        midless_offers_df = midless_offers_df.copy()
        
        if 'merchant_name' in mid_offers_df.columns:
            mid_offers_df['merchant_name_clean'] = mid_offers_df['merchant_name'].apply(
                self.clean_merchant_name
            )
        
        if 'merchant_name' in midless_offers_df.columns:
            midless_offers_df['merchant_name_clean'] = midless_offers_df['merchant_name'].apply(
                self.clean_merchant_name
            )
        
        # Clean location data
        mid_offers_df = self.clean_location_data(mid_offers_df)
        midless_offers_df = self.clean_location_data(midless_offers_df)
        
        # Add source indicator
        mid_offers_df['offer_source'] = 'MID'
        midless_offers_df['offer_source'] = 'MID_LESS'
        
        # Merge datasets
        merged_offers = pd.concat([mid_offers_df, midless_offers_df], ignore_index=True)
        
        # Compute valid_end_date
        merged_offers['valid_end_date'] = merged_offers.apply(
            self.compute_valid_end_date, axis=1
        )
        
        # Parse start dates
        merged_offers['start_date'] = merged_offers['start_date'].apply(self.parse_date)
        
        # Deduplicate based on key fields
        dedup_columns = ['merchant_name_clean', 'zip_code', 'city', 'state', 
                        'reward_type', 'reward_value', 'start_date']
        dedup_columns = [col for col in dedup_columns if col in merged_offers.columns]
        
        merged_offers = merged_offers.drop_duplicates(subset=dedup_columns, keep='first')
        
        return merged_offers
    
    def mid_based_matching(self, transactions_df: pd.DataFrame, 
                          offers_df: pd.DataFrame) -> List[Dict]:
        """
        Perform MID-based matching with high confidence.
        
        Args:
            transactions_df: Cleaned transactions DataFrame
            offers_df: Cleaned offers DataFrame
            
        Returns:
            List of match dictionaries
        """
        matches = []
        
        # Filter offers that have MID
        mid_offers = offers_df[
            (offers_df['offer_source'] == 'MID') & 
            (~pd.isna(offers_df.get('mid')))
        ].copy()
        
        if mid_offers.empty:
            return matches
        
        for _, tx in transactions_df.iterrows():
            if pd.isna(tx.get('mid')):
                continue
            
            tx_date = self.parse_date(tx.get('transaction_date'))
            if not tx_date:
                continue
            
            # Find offers with matching MID
            matching_offers = []
            
            for _, offer in mid_offers.iterrows():
                if (str(tx['mid']) == str(offer['mid']) and
                    self.location_matches(tx, offer) and
                    self.date_in_range(tx_date, offer['start_date'], offer['valid_end_date'])):
                    
                    matching_offers.append({
                        'offer_id': offer['offer_id'],
                        'reward_type': offer['reward_type'],
                        'reward_value': offer['reward_value'],
                        'merchant_name': offer.get('merchant_name', ''),
                        'match_reason': f"MID match: {tx['mid']}"
                    })
            
            # Select best offer
            if matching_offers:
                best_offer = self.select_best_offer(matching_offers)
                matches.append({
                    'tx_id': tx['tx_id'],
                    'offer_id': best_offer['offer_id'],
                    'match_type': 'MID_BASED',
                    'reward_value': best_offer['reward_value'],
                    'reason': best_offer['match_reason']
                })
        
        return matches
    
    def midless_matching(self, transactions_df: pd.DataFrame, 
                        offers_df: pd.DataFrame) -> List[Dict]:
        """
        Perform MID-less matching using fuzzy name matching.
        
        Args:
            transactions_df: Cleaned transactions DataFrame
            offers_df: Cleaned offers DataFrame
            
        Returns:
            List of match dictionaries
        """
        matches = []
        
        # Filter MID-less offers
        midless_offers = offers_df[offers_df['offer_source'] == 'MID_LESS'].copy()
        
        if midless_offers.empty:
            return matches
        
        for _, tx in transactions_df.iterrows():
            tx_date = self.parse_date(tx.get('transaction_date'))
            if not tx_date:
                continue
            
            tx_merchant_clean = self.clean_merchant_name(tx.get('merchant_name', ''))
            if not tx_merchant_clean:
                continue
            
            matching_offers = []
            
            for _, offer in midless_offers.iterrows():
                offer_merchant_clean = offer.get('merchant_name_clean', '')
                if not offer_merchant_clean:
                    continue
                
                # Fuzzy matching with token_sort_ratio
                similarity_score = fuzz.token_sort_ratio(tx_merchant_clean, offer_merchant_clean)
                
                if (similarity_score >= 90 and
                    self.location_matches(tx, offer) and
                    self.date_in_range(tx_date, offer['start_date'], offer['valid_end_date'])):
                    
                    matching_offers.append({
                        'offer_id': offer['offer_id'],
                        'reward_type': offer['reward_type'],
                        'reward_value': offer['reward_value'],
                        'merchant_name': offer.get('merchant_name', ''),
                        'match_reason': f"Fuzzy match: {similarity_score}% similarity"
                    })
            
            # Select best offer
            if matching_offers:
                best_offer = self.select_best_offer(matching_offers)
                matches.append({
                    'tx_id': tx['tx_id'],
                    'offer_id': best_offer['offer_id'],
                    'match_type': 'MIDLESS',
                    'reward_value': best_offer['reward_value'],
                    'reason': best_offer['match_reason']
                })
        
        return matches
    
    def process_transactions_and_offers(self, transactions_df: pd.DataFrame,
                                      mid_offers_df: pd.DataFrame,
                                      midless_offers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing function to match transactions with offers.
        
        Args:
            transactions_df: Raw transactions data
            mid_offers_df: Raw MID-based offers data
            midless_offers_df: Raw MID-less offers data
            
        Returns:
            DataFrame with matched transactions
        """
        # Clean transaction data
        transactions_clean = transactions_df.copy()
        transactions_clean['merchant_name_clean'] = transactions_clean['merchant_name'].apply(
            self.clean_merchant_name
        )
        transactions_clean = self.clean_location_data(transactions_clean)
        
        # Clean and merge offers
        offers_clean = self.clean_and_merge_offers(mid_offers_df, midless_offers_df)
        
        # Perform MID-based matching first (higher confidence)
        mid_matches = self.mid_based_matching(transactions_clean, offers_clean)
        
        # Get transaction IDs that were already matched
        matched_tx_ids = {match['tx_id'] for match in mid_matches}
        
        # Perform MID-less matching on remaining transactions
        remaining_transactions = transactions_clean[
            ~transactions_clean['tx_id'].isin(matched_tx_ids)
        ]
        midless_matches = self.midless_matching(remaining_transactions, offers_clean)
        
        # Combine all matches
        all_matches = mid_matches + midless_matches
        
        # Convert to DataFrame
        if all_matches:
            results_df = pd.DataFrame(all_matches)
        else:
            results_df = pd.DataFrame(columns=['tx_id', 'offer_id', 'match_type', 
                                             'reward_value', 'reason'])
        
        return results_df


# Example usage and testing
def create_sample_data():
    """Create sample data for testing the matching system."""
    
    # Sample transactions
    transactions = pd.DataFrame({
        'tx_id': ['tx_001', 'tx_002', 'tx_003', 'tx_004', 'tx_005'],
        'merchant_name': ['STARBUCKS COFFEE', 'TARGET STORE', 'SHELL GAS STATION', 
                         'MCDONALDS #123', 'WALMART SUPERCENTER'],
        'mid': ['MID001', 'MID002', None, 'MID004', None],
        'transaction_date': ['2024-01-15', '2024-01-20', '2024-01-25', '2024-02-01', '2024-02-05'],
        'zip_code': ['10001', '90210', '60601', '30301', '94102'],
        'city': ['New York', 'Beverly Hills', 'Chicago', 'Atlanta', 'San Francisco'],
        'state': ['NY', 'CA', 'IL', 'GA', 'CA'],
        'amount': [15.50, 75.25, 45.00, 12.75, 125.50]
    })
    
    # Sample MID-based offers
    mid_offers = pd.DataFrame({
        'offer_id': ['offer_001', 'offer_002', 'offer_003'],
        'merchant_name': ['Starbucks', 'Target Corporation', 'Shell'],
        'mid': ['MID001', 'MID002', 'MID003'],
        'reward_type': ['percentage', 'fixed', 'percentage'],
        'reward_value': [5.0, 10.0, 3.0],
        'start_date': ['2024-01-01', '2024-01-15', '2024-01-01'],
        'expiration_date': ['2024-03-31', '2024-02-28', '2024-04-30'],
        'deleted_at': [None, None, None],
        'zip_code': ['10001', '90210', '60601'],
        'city': ['New York', 'Beverly Hills', 'Chicago'],
        'state': ['NY', 'CA', 'IL']
    })
    
    # Sample MID-less offers
    midless_offers = pd.DataFrame({
        'offer_id': ['offer_004', 'offer_005', 'offer_006'],
        'merchant_name': ['McDonalds Restaurant', 'Walmart Inc', 'Starbucks Coffee Co'],
        'reward_type': ['percentage', 'fixed', 'percentage'],
        'reward_value': [8.0, 15.0, 6.0],
        'start_date': ['2024-01-01', '2024-02-01', '2024-01-01'],
        'expiration_date': ['2024-06-30', '2024-03-31', '2024-12-31'],
        'deleted_at': [None, None, None],
        'zip_code': ['30301', '94102', '10001'],
        'city': ['Atlanta', 'San Francisco', 'New York'],
        'state': ['GA', 'CA', 'NY']
    })
    
    return transactions, mid_offers, midless_offers


def main():
    """Main function to demonstrate the matching system."""
    
    # Create sample data
    transactions_df, mid_offers_df, midless_offers_df = create_sample_data()
    
    print("=== Credit Card Transaction Offer Matching System ===\n")
    print("Sample Transactions:")
    print(transactions_df.to_string(index=False))
    print("\nSample MID-based Offers:")
    print(mid_offers_df.to_string(index=False))
    print("\nSample MID-less Offers:")
    print(midless_offers_df.to_string(index=False))
    
    # Initialize matcher
    matcher = TransactionOfferMatcher(proximity_radius_miles=25.0)
    
    # Process matches
    results = matcher.process_transactions_and_offers(
        transactions_df, mid_offers_df, midless_offers_df
    )
    
    print("\n=== MATCHING RESULTS ===")
    if not results.empty:
        print(results.to_string(index=False))
        
        # Save to CSV
        results.to_csv('matched_transactions.csv', index=False)
        print(f"\nResults saved to 'matched_transactions.csv'")
        print(f"Total matches found: {len(results)}")
        print(f"MID-based matches: {len(results[results['match_type'] == 'MID_BASED'])}")
        print(f"MID-less matches: {len(results[results['match_type'] == 'MIDLESS'])}")
    else:
        print("No matches found.")


if __name__ == "__main__":
    main()