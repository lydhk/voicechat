# Download a .txt file from GitHub Pages and save it locally

import requests

def main():
	url = "https://lydhk.github.io/voicechat-data/menu.txt"  # Replace with your actual URL
	local_filename = "context.txt"  # Replace with your desired filename

	response = requests.get(url)
	if response.status_code == 200:
		with open(local_filename, "w", encoding="utf-8") as f:
			f.write(response.text)
		print(f"Downloaded {local_filename} successfully.")
	else:
		print(f"Failed to download file. Status code: {response.status_code}")

if __name__ == "__main__":
	main()
