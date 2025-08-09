This is a comprehensive Python solution that meets all your requirements. This will be a modular system for matching credit card transactions with offers using both MID-based and MID-less approaches.I've created a comprehensive Python solution that meets all your requirements. Here's what the system includes:

## Key Features

### 1. **Data Cleaning & Normalization**
- `clean_merchant_name()`: Removes common business suffixes, normalizes text
- `clean_location_data()`: Standardizes ZIP codes, cities, and states
- `parse_date()`: Handles multiple date formats robustly

### 2. **Offer Dataset Management**
- `clean_and_merge_offers()`: Merges MID and MID-less datasets with deduplication
- `compute_valid_end_date()`: Implements business rules for end date calculation:
  - Uses `expiration_date` if `deleted_at` is null
  - Uses `deleted_at` if `expiration_date` is null  
  - Uses earlier date if both exist
  - Defaults to 2099-12-31 if neither exists

### 3. **Two Matching Pipelines**

**MID-Based Matching (High Confidence)**:
- Exact MID matching
- Location validation (ZIP or city+state)
- Date range validation
- Best reward selection

**MID-Less Matching (Lower Confidence)**:
- Fuzzy merchant name matching (≥90% similarity using `token_sort_ratio`)
- ZIP proximity matching using Haversine distance
- Location and date validation
- Best reward selection

### 4. **ZIP-to-Lat/Long Proximity**
- `haversine_distance()`: Calculates distances in miles
- `is_zip_within_radius()`: Configurable radius matching
- Caching system for coordinate lookups

### 5. **Best Offer Selection**
- Prioritizes percentage rewards over fixed rewards
- Within same type, selects highest value
- `calculate_reward_priority()` and `select_best_offer()` handle the logic

### 6. **Modular Architecture**
- Clean separation of concerns with dedicated helper functions
- Configurable proximity radius
- Comprehensive error handling and validation

## Usage

The system processes your data in this order:
1. Clean and normalize all input data
2. Merge and deduplicate offer datasets  
3. Run MID-based matching first (higher confidence)
4. Run MID-less matching on remaining transactions
5. Output results with match type, reward value, and reasoning

## Output Format

The CSV output includes:
- `tx_id`: Transaction identifier
- `offer_id`: Matched offer identifier  
- `match_type`: 'MID_BASED' or 'MIDLESS'
- `reward_value`: Best reward value found
- `reason`: Detailed matching explanation

## Customization

You can easily customize:
- Proximity radius (default 25 miles)
- Fuzzy matching threshold (default 90%)
- Date formats and business rules
- Reward prioritization logic

The sample data demonstrates typical matching scenarios. To use with your real data, simply replace the sample DataFrames with your actual transaction and offer data.# robustmatching
