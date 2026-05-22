import os
import urllib.request
import zipfile

def download_and_extract_cmapss():
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip"
    zip_path = os.path.join(data_dir, 'CMAPSSData.zip')
    
    print("Downloading CMAPSS dataset from NASA...")
    try:
        # Mask the Python script as a standard web browser to bypass bot-blocking
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
            
        print("Download complete. Extracting files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        print(f"Extraction complete. Data is ready in: {data_dir}")
        
        # Clean up the zip file to save space
        os.remove(zip_path)
        
    except Exception as e:
        print(f"Failed to download or extract data: {e}")

if __name__ == "__main__":
    download_and_extract_cmapss()