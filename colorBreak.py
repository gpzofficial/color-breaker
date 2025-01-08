import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean

def extract_unique_colors(image_path):
    # Open and process image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get image dimensions
    width, height = img.size
    
    # Convert to numpy array and reshape
    pixel_array = np.array(img)
    pixels = pixel_array.reshape(-1, 3)
    
    # Get unique colors and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # Sort by frequency (descending)
    sort_indices = np.argsort(-counts)
    unique_colors = unique_colors[sort_indices]
    counts = counts[sort_indices]
    
    # Save the dimensions along with color data
    color_data = np.column_stack((unique_colors, counts))
    np.save('color_frequencies.npy', color_data)
    
    return color_data, (height, width)

def rearrange_colors(color_data):
    colors = color_data[:, :3]
    frequencies = color_data[:, 3]
    total_colors = len(colors)
    
    # Initialize arrays
    rearranged_colors = np.zeros((total_colors, 3))
    rearranged_frequencies = np.zeros(total_colors)
    used_indices = np.zeros(total_colors, dtype=bool)
    
    # Start with most frequent color
    start_idx = np.argmax(frequencies)
    rearranged_colors[0] = colors[start_idx]
    rearranged_frequencies[0] = frequencies[start_idx]
    used_indices[start_idx] = True
    
    # Process remaining colors
    for i in range(1, total_colors):
        if i % 100 == 0:
            print(f"Processing color {i}/{total_colors}")
        
        current_color = rearranged_colors[i-1].reshape(1, -1)
        unused_mask = ~used_indices
        
        # Calculate distances using only RGB values
        distances = np.sqrt(np.sum((colors[unused_mask] - current_color) ** 2, axis=1))
        
        # Find nearest color
        nearest_idx = np.argmin(distances)
        original_idx = np.where(unused_mask)[0][nearest_idx]
        
        # Update arrays
        rearranged_colors[i] = colors[original_idx]
        rearranged_frequencies[i] = frequencies[original_idx]
        used_indices[original_idx] = True
    
    # Combine and save results
    result = np.column_stack((rearranged_colors, rearranged_frequencies))
    np.save('rearranged_colors.npy', result)
    
    return result

def generate_sequential_image(color_data_path, dimensions, output_path):
    # Load the rearranged color data with frequencies
    color_data = color_data_path
    height, width = dimensions
    
    # Create empty image array
    reconstructed = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill image sequentially based on frequencies
    pixel_idx = 0
    
    # For each color and its frequency
    for color_entry in color_data:
        # Get color and frequency (corrected indexing)
        color = color_entry[:3].astype(np.uint8)  # First three values are RGB
        freq = int(color_entry[3])  # Fourth value is frequency
        
        # Place this color 'freq' times
        for _ in range(freq):
            if pixel_idx >= height * width:
                break
                
            y = pixel_idx // width
            x = pixel_idx % width
            
            reconstructed[y, x] = color
            pixel_idx += 1

    
    # Print debug information
    print(f"Image dimensions: {width}x{height} = {width*height} pixels")
    print(f"Total pixels filled: {pixel_idx}")
    print(f"Total frequencies sum: {int(np.sum(color_data[:, 3]))}")
    
    # Save the image
    img = Image.fromarray(reconstructed)
    img.save(output_path)
    return reconstructed

def main():
    # Input image path
    input_path = "src.png"
    
    # Step 1: Extract unique colors and their frequencies
    print("Extracting unique colors...")
    color_data, dimensions = extract_unique_colors(input_path)
    print(f"Found {len(color_data)} unique colors")
    
    # Step 2: Rearrange colors based only on RGB similarity
    print("\nRearranging colors...")
    rearranged_data = rearrange_colors(color_data)
    
    # Step 3: Generate new image
    print("\nGenerating new image...")
    output_path = "out.png"
    reconstructed = generate_sequential_image(rearranged_data, dimensions, output_path)
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main()
