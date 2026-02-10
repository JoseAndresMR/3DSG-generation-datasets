import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

# Get the benchmark root directory (parent of this script's parent)
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks"

method_order = ["Rand", "EC-square", "EC-L", "Ours", "GT"]


def get_available_methods():
    """Get list of all available methods in the benchmarks folder."""
    methods = [d.name for d in BENCHMARK_DIR.iterdir() if d.is_dir()]
    return sorted(methods)


def get_available_subdatasets():
    """Get all available subdatasets across all methods."""
    subdatasets = set()
    for method_dir in BENCHMARK_DIR.iterdir():
        if method_dir.is_dir():
            for dataset_dir in method_dir.iterdir():
                if dataset_dir.is_dir():
                    for subdataset_dir in dataset_dir.iterdir():
                        if subdataset_dir.is_dir():
                            subdatasets.add(subdataset_dir.name)
    return sorted(subdatasets)


def get_methods_for_subdataset(subdataset):
    """
    Get all methods that contain a specific subdataset, sorted by method_order.
    
    Args:
        subdataset: Subdataset name (e.g., 'M-F')
    
    Returns:
        List of method names that contain this subdataset, ordered by method_order
    """
    methods_with_subdataset = []
    for method_dir in BENCHMARK_DIR.iterdir():
        if method_dir.is_dir():
            method_name = method_dir.name
            for dataset_dir in method_dir.iterdir():
                if dataset_dir.is_dir():
                    subdataset_path = dataset_dir / subdataset
                    if subdataset_path.exists():
                        methods_with_subdataset.append(method_name)
                        break
    
    # Sort by method_order, with unlisted methods at the end
    def sort_key(method):
        try:
            return method_order.index(method)
        except ValueError:
            return len(method_order)
    
    return sorted(methods_with_subdataset, key=sort_key)


def count_available_examples(method, subdataset):
    """
    Count how many examples are available for a given method and subdataset.
    
    Args:
        method: Method name
        subdataset: Subdataset name
    
    Returns:
        Number of available examples
    """
    count = 0
    method_dir = BENCHMARK_DIR / method
    if not method_dir.exists():
        return count
    
    for dataset_dir in method_dir.iterdir():
        if dataset_dir.is_dir():
            subdataset_path = dataset_dir / subdataset
            if subdataset_path.exists():
                # Count graph_*.png files
                for img_file in subdataset_path.glob("graph_*_3d.png"):
                    count += 1
                return count
    
    return count


def get_image_path(method, subdataset, example_idx):
    """
    Get the image path for a specific method, subdataset, and example.
    
    Args:
        method: Method name (e.g., 'EC-L', 'EC-square')
        subdataset: Subdataset name (e.g., 'M-F', 'R-B')
        example_idx: Example index (0-4)
    
    Returns:
        Path object if image exists, None otherwise
    """
    # Try to find the image by traversing datasets
    method_dir = BENCHMARK_DIR / method
    if not method_dir.exists():
        return None
    
    # Search through all datasets for the subdataset
    for dataset_dir in method_dir.iterdir():
        if dataset_dir.is_dir():
            subdataset_path = dataset_dir / subdataset
            if subdataset_path.exists():
                image_path = subdataset_path / f"graph_{example_idx}_3d.png"
                if image_path.exists():
                    return image_path
    
    return None


def compare_methods_on_subdataset(subdataset, methods=None, 
                                  figsize=None, save_path=None):
    """
    Create a comparison plot of methods for a specific subdataset.
    
    Args:
        subdataset: Subdataset name (e.g., 'M-F')
        methods: List of methods to compare. If None, uses all available methods.
        figsize: Figure size tuple (width, height). If None, auto-calculated.
        save_path: Path to save the figure. If None, only displays.
    
    Returns:
        (fig, ax) matplotlib figure and axes objects
    """
    if methods is None:
        methods = get_available_methods()
    
    # Determine the actual number of examples by checking the first method
    num_examples = 0
    if methods:
        num_examples = count_available_examples(methods[0], subdataset)
    
    if num_examples == 0:
        num_examples = 5  # Default fallback
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (num_examples * 2.5, len(methods) * 2.5)
    
    # Create figure with subplots (rows=methods, columns=examples)
    fig, axes = plt.subplots(len(methods), num_examples, figsize=figsize)
    
    # Ensure axes is always 2D numpy array
    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            if len(methods) == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.reshape(-1, 1)
    else:
        # axes is a single Axes object (1x1 case)
        axes = np.array([[axes]])
    
    fig.suptitle(f"Dataset: {subdataset}", fontsize=16, fontweight='bold')
    
    # Load and display images
    for row_idx, method in enumerate(methods):
        for col_idx in range(num_examples):
            ax = axes[row_idx, col_idx]
            
            # Get image path
            img_path = get_image_path(method, subdataset, col_idx)
            
            if img_path and img_path.exists():
                # Load and display image
                img = imread(img_path)
                ax.imshow(img)
            else:
                # Show placeholder if image not found
                ax.text(0.5, 0.5, 'Image not found', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red')
            
            # Set titles and remove axes
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add row labels (method names) on left side
        axes[row_idx, 0].set_ylabel(method, fontsize=11, fontweight='bold', 
                                   labelpad=10)
    
    # Add "examples" label at the bottom center spanning all columns
    fig.text(0.5, 0.02, 'examples', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def compare_subdatasets_across_methods(subdatasets, method, num_examples=5,
                                       figsize=None, save_path=None):
    """
    Create a comparison plot of subdatasets for a specific method.
    
    Args:
        subdatasets: List of subdataset names to compare
        method: Method name to use
        num_examples: Number of examples to display (columns)
        figsize: Figure size tuple (width, height). If None, auto-calculated.
        save_path: Path to save the figure. If None, only displays.
    
    Returns:
        (fig, ax) matplotlib figure and axes objects
    """
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (num_examples * 2.5, len(subdatasets) * 2.5)
    
    # Create figure with subplots (rows=subdatasets, columns=examples)
    fig, axes = plt.subplots(len(subdatasets), num_examples, figsize=figsize)
    
    # Ensure axes is always 2D even with single subdataset
    if len(subdatasets) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Subdataset Comparison: {method}", fontsize=16, fontweight='bold')
    
    # Load and display images
    for row_idx, subdataset in enumerate(subdatasets):
        for col_idx in range(num_examples):
            ax = axes[row_idx, col_idx]
            
            # Get image path
            img_path = get_image_path(method, subdataset, col_idx)
            
            if img_path and img_path.exists():
                # Load and display image
                img = imread(img_path)
                ax.imshow(img)
            else:
                # Show placeholder if image not found
                ax.text(0.5, 0.5, 'Image not found', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red')
            
            # Set titles and remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add column headers (example index) on first row
            if row_idx == 0:
                ax.set_title(f"Example {col_idx}", fontsize=10, fontweight='bold')
        
        # Add row labels (subdataset names) on left side
        axes[row_idx, 0].set_ylabel(subdataset, fontsize=11, fontweight='bold',
                                   labelpad=10)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def sweep_and_generate_comparisons(output_dir="comparison_images"):
    """
    Sweep through all subdatasets in the benchmarks folder and create 
    comparison plots for each subdataset across all methods that contain it.
    
    Args:
        output_dir: Output directory at the same level as benchmarks (default: 'comparison_images')
    """
    output_path = BENCHMARK_DIR.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subdatasets
    subdatasets = get_available_subdatasets()
    
    print(f"Found {len(subdatasets)} subdatasets")
    print(f"Output directory: {output_path}\n")
    
    # Generate comparison for each subdataset
    for subdataset in subdatasets:
        methods = get_methods_for_subdataset(subdataset)
        
        if not methods:
            print(f"⚠️  Skipping {subdataset}: no methods found")
            continue
        
        print(f"Creating comparison for '{subdataset}' with {len(methods)} methods: {methods}")
        
        try:
            # Generate comparison plot
            save_file = output_path / f"{subdataset}_comparison.png"
            fig, axes = compare_methods_on_subdataset(
                subdataset=subdataset,
                methods=methods,
                save_path=str(save_file)
            )
            plt.close(fig)  # Close figure to free memory
            print(f"✓ Saved: {save_file}")
        
        except Exception as e:
            print(f"✗ Error creating comparison for {subdataset}: {e}")
    
    print(f"\n✓ All comparisons completed! Output saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Available methods:", get_available_methods())
    print("Available subdatasets:", get_available_subdatasets())
    print()
    
    # Sweep and generate all comparisons
    sweep_and_generate_comparisons(output_dir="comparison_images")
