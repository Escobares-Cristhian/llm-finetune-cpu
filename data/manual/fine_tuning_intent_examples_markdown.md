# Fine-tuning intent classification examples

## Self-contained summary for fine-tuning intent classification

The model should receive natural-language geospatial/remote-sensing queries and respond only with JSON matching this schema:

```json
{
  "id": "<UUID version 7>",
  "method_to_use": "<describe-globally | describe-with-filter | create | edit | rerun>"
}
```

The model is **not selecting a geoprocessing tool** such as `rgb_single`, `index_composite`, or `bands_single`. It is only detecting the user’s **workflow intent**.

Use these intent labels:

| `method_to_use` | Meaning |
|---|---|
| `describe-globally` | The user wants a general explanation, overview, recommendation, comparison, or help understanding what is possible. No concrete output/job should be created. |
| `describe-with-filter` | The user asks for explanation or analysis constrained by filters such as product, bbox, dates, bands, reducer, resolution, cloud masking, index type, or region. Still descriptive, not creating or modifying a job. |
| `create` | The user wants to generate a new geospatial output/job/map/export/composite/index/RGB/band stack. Often includes product, bbox, date or date range, bands, reducer, ND bands, resolution, cloud mask, or scale offset. |
| `edit` | The user wants to modify a previous or existing request/job/output: change date, bbox, bands, reducer, cloud masking, resolution, projection, tile size, max tiles, product, etc. Follow-up wording may be short: “make it median,” “use Sentinel-2 instead,” “crop it tighter.” |
| `rerun` | The user wants to repeat the previous job/request, either exactly or with wording like “run it again,” “retry,” “rerun the same export,” “generate it again.” Minor non-substantive retries remain `rerun`; substantive parameter changes are `edit`. |

Queries can appear as a **first message** with full context, or as a **mid-conversation message** that refers to earlier context using phrases like “same area,” “that one,” “use the previous bbox,” “again,” “change it,” or “now do July.” Mid-conversation examples may be ambiguous without history, but the surface intent should still be classified.

---

## 10 example queries as first message

### 1

User query:

> What kinds of satellite products can I use to visualize vegetation health over a farm?

Expected response:

```json
{
  "id": "019eaf54-3b0a-7efc-a813-7a626afe8b08",
  "method_to_use": "describe-globally"
}
```

### 2

User query:

> Explain the difference between an RGB composite and an NDVI export for Sentinel-2.

Expected response:

```json
{
  "id": "019eaf54-3b0a-73ca-8785-e237fef50498",
  "method_to_use": "describe-globally"
}
```

### 3

User query:

> For Sentinel-2 SR in bbox -58.55,-34.75,-58.30,-34.55 between 2024-01-01 and 2024-03-31, what reducer would be best if I want to avoid clouds?

Expected response:

```json
{
  "id": "019eaf54-3b0a-7ff8-ab5f-ca701413b155",
  "method_to_use": "describe-with-filter"
}
```

### 4

User query:

> Describe what bands B4, B3, and B2 would show for COPERNICUS/S2_SR_HARMONIZED on 2023-11-15 around -70.75,-33.65,-70.45,-33.35.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7be3-bc0d-1ace71861a79",
  "method_to_use": "describe-with-filter"
}
```

### 5

User query:

> Create a true-color Sentinel-2 RGB GeoTIFF for bbox -60.8,-32.2,-60.4,-31.9 on 2024-02-10 using bands B4,B3,B2.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7d36-976e-1089dc4fcf4c",
  "method_to_use": "create"
}
```

### 6

User query:

> Generate a median NDVI composite from Landsat 8 Collection 2 Level 2 for -64.35,-31.55,-64.05,-31.30 from 2023-10-01 to 2023-12-31.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7bfa-8869-560fad82a2e7",
  "method_to_use": "create"
}
```

### 7

User query:

> Export all available bands from MODIS for bbox -63,-35,-62,-34 on 2022-08-20 at native resolution.

Expected response:

```json
{
  "id": "019eaf54-3b0a-709f-912d-cd1058f28536",
  "method_to_use": "create"
}
```

### 8

User query:

> Build a false-color composite with Sentinel-2 B8,B4,B3 for 2024-05-01 to 2024-05-31, bbox -59.1,-34.2,-58.7,-33.9, using median and cloud mask.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7477-9729-c28a7c449ffb",
  "method_to_use": "create"
}
```

### 9

User query:

> I already generated the NDVI tiles, but change the resolution to 30 meters and increase max tiles to 40.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7351-b12d-e6bef08a6081",
  "method_to_use": "edit"
}
```

### 10

User query:

> Rerun the same composite export again; the previous download links expired.

Expected response:

```json
{
  "id": "019eaf54-3b0a-77e1-8563-e9f414fe340d",
  "method_to_use": "rerun"
}
```

---

## 10 example queries as mid-conversation message, without the previous message

### 1

User query:

> Same area, but explain what the median reducer would change.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7d10-af4e-b80e36c2cc87",
  "method_to_use": "describe-with-filter"
}
```

### 2

User query:

> What would happen if I used B8 instead of B4 there?

Expected response:

```json
{
  "id": "019eaf54-3b0a-7f53-8b4f-62c811197c44",
  "method_to_use": "describe-with-filter"
}
```

### 3

User query:

> Now create it with cloud masking enabled.

Expected response:

```json
{
  "id": "019eaf54-3b0a-730b-8e9b-db81b96fb9bc",
  "method_to_use": "create"
}
```

### 4

User query:

> Generate the NDVI version for the same bbox and date range.

Expected response:

```json
{
  "id": "019eaf54-3b0a-788a-a9ed-09f797e88605",
  "method_to_use": "create"
}
```

### 5

User query:

> Change it to a mosaic reducer instead of median.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7b34-a248-169e7a80f92c",
  "method_to_use": "edit"
}
```

### 6

User query:

> Use EPSG:4326 and lower the tile size to 1000 pixels.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7709-854a-93d0c3d49e35",
  "method_to_use": "edit"
}
```

### 7

User query:

> Actually use the previous bbox but switch the date to 2024-06-15.

Expected response:

```json
{
  "id": "019eaf54-3b0a-79cc-a4cd-824d660a28bf",
  "method_to_use": "edit"
}
```

### 8

User query:

> Run that again, same parameters.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7014-8fc8-3c0c1b6ef1b8",
  "method_to_use": "rerun"
}
```

### 9

User query:

> Retry the export; I think the first one failed.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7420-ba9e-b086082119d7",
  "method_to_use": "rerun"
}
```

### 10

User query:

> Before running it, summarize what the current parameters mean.

Expected response:

```json
{
  "id": "019eaf54-3b0a-7ef3-90b4-ca7a45f36881",
  "method_to_use": "describe-with-filter"
}
```
