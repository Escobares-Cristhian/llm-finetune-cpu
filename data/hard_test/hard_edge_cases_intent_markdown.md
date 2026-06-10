# Hard edge cases for geospatial intent classification

These examples are intentionally ambiguous. They are useful for fine-tuning because they force the model to separate `describe-globally`, `describe-with-filter`, `create`, `edit`, and `rerun` based on intent rather than tool names.

Tie-break guidance:

- Prefer `describe-with-filter` when the user asks for explanation, comparison, advice, preview, or parameter meaning using an existing or implied context.
- Prefer `edit` when the user changes any substantive parameter of an existing request, even if they say “same thing.”
- Prefer `rerun` only when the user repeats the prior job with no substantive parameter change.
- Prefer `create` when the user defines a new output/job from scratch or asks to set up a new export.

## 10 hard edge cases

### 1

User query: `Can you make the same NDVI again, but first remind me what the current settings are?`

Why hard: It contains both `rerun` and `describe-with-filter`. The latest explicit action is to make it again, but the user asks for a reminder first.

Expected response: `{"id":"019eaf5f-e65a-74b4-adac-2a9557384714","method_to_use":"rerun"}`

### 2

User query: `Use the previous bbox and tell me whether median or mosaic is better before exporting.`

Why hard: It mentions a future export, but the current request is a decision/explanation constrained by existing filters.

Expected response: `{"id":"019eaf5f-e65a-7efa-88c2-a85d7d1f5127","method_to_use":"describe-with-filter"}`

### 3

User query: `Actually, don't run it yet; just set it up with Sentinel-2 B8,B4,B3 for July.`

Why hard: `set it up` sounds like preparing a new output, while `don't run it yet` avoids execution language. For workflow routing, this is still creation intent because parameters for a new job are being specified.

Expected response: `{"id":"019eaf5f-e65a-77a0-8a1f-b926f7940d31","method_to_use":"create"}`

### 4

User query: `Can we try 20 meter resolution instead?`

Why hard: Short mid-conversation request with no object named. It likely modifies the previous job/output.

Expected response: `{"id":"019eaf5f-e65a-7547-922e-4de1044597c3","method_to_use":"edit"}`

### 5

User query: `Do the same thing, but for August instead of July.`

Why hard: `same thing` suggests `rerun`, but changing the date is a substantive parameter change.

Expected response: `{"id":"019eaf5f-e65a-7307-ba0a-ba609c71bc23","method_to_use":"edit"}`

### 6

User query: `Run it again, and if it still fails use fewer tiles.`

Why hard: The primary request is `rerun`, but it includes a conditional modification. Since the modification only applies after failure, classify the immediate intent as rerun.

Expected response: `{"id":"019eaf5f-e65a-7aa0-916b-a1848ed3cdf5","method_to_use":"rerun"}`

### 7

User query: `I need a cloud-free view of this area; should I use a single date or a composite?`

Why hard: The user has a goal and maybe a region from prior context, but asks for advice rather than generation.

Expected response: `{"id":"019eaf5f-e65a-7390-ab51-08b72481a3b6","method_to_use":"describe-with-filter"}`

### 8

User query: `Make the export smaller so it doesn't exceed the tile cap.`

Why hard: No specific parameter is named, but the user wants to change the existing output constraints, likely resolution, bbox, tile size, or max tiles.

Expected response: `{"id":"019eaf5f-e65a-7640-8ed3-15b2749b637f","method_to_use":"edit"}`

### 9

User query: `Give me the RGB for the same place, not NDVI.`

Why hard: It could be interpreted as creating a new RGB product or editing the previous NDVI request. Because it replaces the output type of the previous request, classify as edit.

Expected response: `{"id":"019eaf5f-e65a-7d52-b1f9-023dc180da65","method_to_use":"edit"}`

### 10

User query: `What would the output look like if I applied scale and offset?`

Why hard: It references a processing option and output behavior, but asks hypothetically rather than requesting a changed export.

Expected response: `{"id":"019eaf5f-e65a-7cc6-911b-e3b724349019","method_to_use":"describe-with-filter"}`
