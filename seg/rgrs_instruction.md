# Rule-guided preserved-region localization

## Task

Identify complete foreground objects or coherent foreground scene regions with natural boundaries that can be preserved while the surrounding content is transformed. Return candidate bounding boxes for those regions. The downstream program will choose a candidate using a shared spatial reference point; you only localize candidates.

## Localization rules

- Ground every candidate in the visible image.
- Describe each distinct candidate once using a short English label.
- Include the full visible extent of an object. Do not return a wheel, head, limb, texture patch, or another part when its complete object is visible.
- If multiple separate foreground objects are visible, return separate candidates rather than one box covering unrelated objects and the background between them.
- A coherent foreground scene region is allowed when it has clear natural boundaries.
- Do not create candidates for cast shadows, reflections, image borders, or the complete image.
- Do not infer hidden content, invent object categories, or invent object boundaries.
- Return at least one and at most {{max_candidates}} candidates.
- Candidate order does not express priority. Do not select a candidate yourself.

## Coordinate system

The image provided to you is exactly {{image_width}} pixels wide and {{image_height}} pixels high. Use absolute pixel coordinates in this provided image, not coordinates normalized to 0-1 or 0-1000. The origin is the top-left corner, x increases to the right, and y increases downward.

Each `bbox_2d` must be `[x1, y1, x2, y2]`, where `x1 < x2` and `y1 < y2`. All values must be finite JSON numbers. Both x coordinates must lie between 0 and {{image_width}}, and both y coordinates must lie between 0 and {{image_height}}. Draw tight boxes around the complete visible regions.

## Output format

Return exactly one JSON object and no Markdown, explanation, reasoning, or additional text. Follow this JSON Schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["candidates"],
  "properties": {
    "candidates": {
      "type": "array",
      "minItems": 1,
      "maxItems": {{max_candidates}},
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["label", "bbox_2d"],
        "properties": {
          "label": {
            "type": "string",
            "minLength": 1
          },
          "bbox_2d": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
              "type": "number"
            }
          }
        }
      }
    }
  }
}
```

The schema specifies the required output structure. Return a candidates object containing your actual labels and coordinates, not the schema itself.
