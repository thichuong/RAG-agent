
import requests
import json
import os

def download_and_process_mappings():
    sources = [
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_full_tickers.json"
    ]
    
    combined_mapping = {}
    
    # 1. Add Crypto Defaults (since stock lists miss these)
    combined_mapping.update({
        "BITCOIN": "BTC-USD", "BTC": "BTC-USD",
        "ETHEREUM": "ETH-USD", "ETH": "ETH-USD",
        "DOGECOIN": "DOGE-USD", "DOGE": "DOGE-USD",
        "SOLANA": "SOL-USD", "SOL": "SOL-USD",
        "CARDANO": "ADA-USD", "ADA": "ADA-USD",
        "RIPPLE": "XRP-USD", "XRP": "XRP-USD",
        "CHAINLINK": "LINK-USD", "LINK": "LINK-USD",
        "BINANCE COIN": "BNB-USD", "BNB": "BNB-USD",
        "LITECOIN": "LTC-USD", "LTC": "LTC-USD",
        "POLKADOT": "DOT-USD", "DOT": "DOT-USD"
    })

    print("Downloading ticker lists...")
    for url in sources:
        try:
            print(f"Fetching {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Expected format: list of dicts with 'symbol' and 'name'
                for item in data:
                    symbol = item.get('symbol')
                    name = item.get('name')
                    if symbol and name:
                        # Normalize name: Uppercase, strip
                        clean_name = name.upper().strip()
                        # Add direct mapping
                        combined_mapping[clean_name] = symbol
                        
                        # Add specific overrides for common short names if they contain the full name
                        # e.g. "APPLE INC." -> "APPLE" mapping
                        # Consolidate replacements
                        replacements = [" INC.", " CORP.", " LTD.", " PLC", " CORPORATION", " COMPANY", " GROUP", " HOLDINGS", " LIMITED", " (THE)", " COMMON STOCK", " CLASS A", " CLASS B", " ORDINARY SHARES", " COMMON SHARES", " AMERICAN DEPOSITARY SHARES", " ADS"]
                        simplified_name = clean_name
                        for r in replacements:
                            simplified_name = simplified_name.replace(r, "")
                        simplified_name = simplified_name.strip()
                        
                        if simplified_name and simplified_name != clean_name:
                             combined_mapping[simplified_name] = symbol
                             
                        # Special handle for just the first word if it represents the main company name and is unique enough (heuristic)
                        # Maybe too risky for general automation, but "APPLE" from "APPLE INC..." is desirable.
                        # Let's rely on the manual overrides for the big ones if this fails, but the improved stripping should work for APPLE COMMON STOCK -> APPLE.
            else:
                print(f"Failed to fetch {url}: {response.status_code}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

    # Add common tech giants explicitly if not caught by heuristics (to be safe)
    # The heuristic above usually catches "APPLE INC." -> "APPLE"
    overrides = {
        "GOOGLE": "GOOGL",
        "NVIDIA": "NVDA",
        "META": "META",
        "FACEBOOK": "META"
    }
    combined_mapping.update(overrides)

    output_path = "src/mapping_data.json"
    with open(output_path, 'w') as f:
        json.dump(combined_mapping, f, indent=2)
    
    print(f"Successfully saved {len(combined_mapping)} mappings to {output_path}")

if __name__ == "__main__":
    download_and_process_mappings()
