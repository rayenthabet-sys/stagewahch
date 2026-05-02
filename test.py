import requests

query = """
[out:json][timeout:25];
area["ISO3166-1"="TN"]->.searchArea;
way["landuse"="orchard"](area.searchArea);
out count;
"""

r = requests.get("https://overpass.kumi.systems/api/interpreter", params={"data": query})
print("Status:", r.status_code)
print("Response:", r.text[:500])