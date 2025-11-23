import urllib.request
import zipfile
import io
import os

url = "https://github.com/nihui/srmd-ncnn-vulkan/releases/download/20220721/srmd-ncnn-vulkan-20220721-windows.zip"
print(f"Downloading {url}...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    z = zipfile.ZipFile(io.BytesIO(response.read()))
    print("Zip contents:")
    for name in z.namelist():
        print(name)
