import json
with open('data/parcelles_OlivierExtensif.json') as f:
    d = json.load(f)
for p in d['parcels']:
    lat = sum(c['lat'] for c in p['coordinates'])/len(p['coordinates'])
    lng = sum(c['lng'] for c in p['coordinates'])/len(p['coordinates'])
    if lat < 30 or lat > 38 or lng < 7 or lng > 12:
        print(f"SUSPICIOUS: {p.get('id')} lat={lat:.3f} lng={lng:.3f}")