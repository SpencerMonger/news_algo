import requests
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def test_url(url, source_name, timeout=10):
    """
    Test if a URL is valid and accessible.
    
    Args:
        url (str): URL to test
        source_name (str): Name of the source for reporting
        timeout (int): Timeout in seconds
        
    Returns:
        dict: Result of the test including status and details
    """
    result = {
        'Source Name': source_name,
        'URL': url,
        'Status': 'Unknown',
        'Status Code': None,
        'Error': None,
        'Redirected URL': None
    }
    
    # Skip URLs that are clearly guesses (based on pattern matching)
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    try:
        # Use a proper user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Make a HEAD request first (faster, less bandwidth)
        response = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        
        # If HEAD request fails, try a GET request
        if response.status_code >= 400:
            response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        
        result['Status Code'] = response.status_code
        
        # Check if redirected
        if response.url != url:
            result['Redirected URL'] = response.url
        
        # Determine status based on status code
        if response.status_code < 400:
            result['Status'] = 'Valid'
        else:
            result['Status'] = 'Error'
            result['Error'] = f"HTTP Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError as e:
        result['Status'] = 'Error'
        result['Error'] = f"Connection Error: {str(e)}"
    except requests.exceptions.Timeout as e:
        result['Status'] = 'Error'
        result['Error'] = f"Timeout Error: {str(e)}"
    except requests.exceptions.TooManyRedirects as e:
        result['Status'] = 'Error'
        result['Error'] = f"Too Many Redirects: {str(e)}"
    except requests.exceptions.RequestException as e:
        result['Status'] = 'Error'
        result['Error'] = f"Request Error: {str(e)}"
    except Exception as e:
        result['Status'] = 'Error'
        result['Error'] = f"Unexpected Error: {str(e)}"
        
    return result

def validate_urls(csv_path, output_path, max_workers=5):
    """
    Validate all URLs in the CSV file and save results.
    
    Args:
        csv_path (str): Path to the CSV file with URLs
        output_path (str): Path to save the validation results
        max_workers (int): Maximum number of concurrent workers
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Loaded {total} URLs to validate")
    
    # Create a list to store results
    results = []
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(test_url, row['URL'], row['Source Name']): (i, row['URL'], row['Source Name']) 
            for i, (_, row) in enumerate(df.iterrows())
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_url)):
            idx, url, source_name = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                status = result['Status']
                status_code = result['Status Code'] if result['Status Code'] else 'N/A'
                error = result['Error'] if result['Error'] else 'None'
                
                status_color = '\033[92m' if status == 'Valid' else '\033[91m'  # Green for valid, red for error
                print(f"[{i+1}/{total}] {status_color}{status}\033[0m - {source_name}: {url} (Code: {status_code}, Error: {error})")
                
                # Save intermediate results every 10 URLs
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(output_path, index=False)
                    print(f"Saved progress to {output_path} ({i+1}/{total} URLs processed)")
                    
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                results.append({
                    'Source Name': source_name,
                    'URL': url,
                    'Status': 'Error',
                    'Status Code': None,
                    'Error': f"Processing Error: {str(e)}",
                    'Redirected URL': None
                })
    
    # Create summary DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    valid_count = len(results_df[results_df['Status'] == 'Valid'])
    error_count = len(results_df[results_df['Status'] == 'Error'])
    
    print("\nValidation Summary:")
    print(f"Total URLs: {total}")
    print(f"Valid URLs: {valid_count} ({valid_count/total*100:.1f}%)")
    print(f"Error URLs: {error_count} ({error_count/total*100:.1f}%)")
    print(f"Results saved to: {output_path}")
    
    # Create a filtered CSV with only the invalid URLs
    invalid_df = results_df[results_df['Status'] != 'Valid']
    invalid_path = output_path.replace('.csv', '_invalid.csv')
    invalid_df.to_csv(invalid_path, index=False)
    print(f"Invalid URLs saved to: {invalid_path}")
    
    return results_df

def main():
    # Set up paths
    logs_dir = r"C:\Users\spenc\Downloads\Dev Files\News_Algo\logs"
    ensure_directory_exists(logs_dir)
    
    input_csv = os.path.join(logs_dir, "news_sources_urls.csv")
    output_csv = os.path.join(logs_dir, "news_sources_urls_validated.csv")
    
    # Validate URLs
    validate_urls(input_csv, output_csv, max_workers=5)

if __name__ == "__main__":
    main() 