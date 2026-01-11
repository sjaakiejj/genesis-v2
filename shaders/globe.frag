#version 120

// Uniforms
uniform sampler2D p3d_Texture0; // Color Texture (stage 0)
uniform sampler2D id_tex;       // Plate ID Texture (custom input)
uniform sampler2D selection_tex;// Selection State (2D 256x1, indexed by ID)
uniform sampler2D vector_tex;   // Vector Arrow Overlay
uniform float show_vectors;     // Toggle visibility (0.0 or 1.0)

// Inputs from Vertex Shader
varying vec2 texcoord;
varying vec3 normal;

void main() {
    // 1. Sample Color
    vec4 color = texture2D(p3d_Texture0, texcoord);

    // 2. Sample Plate ID (0.0 - 1.0 range, scaled from 0-255)
    // Using FTNearest in renderer ensures we get discrete values
    float id_norm = texture2D(id_tex, texcoord).r;
    
    // 3. Check Selection Status
    // Sample the 2D selection texture (256x1).
    // Correctly map the normalized ID (k/255) to the texture coordinate ( (k+0.5)/256 ).
    float sel_coord = (floor(id_norm * 255.0 + 0.5) + 0.5) / 256.0;
    float is_selected = texture2D(selection_tex, vec2(sel_coord, 0.5)).r;

    // 4. Highlight Logic
    if (is_selected > 0.5) {
        // Boost brightness for selected plate
        color.rgb *= 1.4;
    }

    // 5. Border Detection (for all plates - black borders)
    float border = 0.0;
    float width_u = 0.003; // Approx 3 pixels at 1024 width
    float width_v = 0.006; // Approx 3 pixels at 512 height
    
    // Check 4 neighbors
    float id_u1 = texture2D(id_tex, texcoord + vec2(width_u, 0.0)).r;
    float id_u2 = texture2D(id_tex, texcoord - vec2(width_u, 0.0)).r;
    float id_v1 = texture2D(id_tex, texcoord + vec2(0.0, width_v)).r;
    float id_v2 = texture2D(id_tex, texcoord - vec2(0.0, width_v)).r;
    
    // If any neighbor has different ID, it's a border
    if (abs(id_u1 - id_norm) > 0.001 || abs(id_u2 - id_norm) > 0.001 ||
        abs(id_v1 - id_norm) > 0.001 || abs(id_v2 - id_norm) > 0.001) {
        border = 1.0;
    }
    
    // Apply border color
    if (border > 0.5) {
        if (is_selected > 0.5) {
            // White border for selected plates
            color.rgb = vec3(1.0, 1.0, 1.0);
        } else {
            // Black border for all other plates
            color.rgb = vec3(0.0, 0.0, 0.0);
        }
    }

    // Basic lighting (hacky, since we removed auto-shader)
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(normal, lightDir), 0.2);
    color.rgb *= diff;
    
    // 6. Vector Overlay (Unlit)
    vec4 vec_color = texture2D(vector_tex, texcoord);
    color.rgb = mix(color.rgb, vec_color.rgb, vec_color.a * show_vectors);

    gl_FragColor = color;
}
