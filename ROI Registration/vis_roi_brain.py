import subprocess
import os
import nibabel as nib
import pyvista as pv
import meshio
import numpy as np

# File paths
output_directory = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/bobby_head"
msh_path = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/bobby_head/m2m_bobby/bobby.msh"
fixed_image_path = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/bobby_head/bobby_T1.nii.gz"
# transform_affine_path = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/bobby_head/transform_mni_to_bobby0GenericAffine.mat"
transform_affine_path = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/ROI Registration/ccc7_24jan25_r_af_inv.mat"
warp_affine_path = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/ROI Registration/ccc7_24jan25_warp_af_inv.mat"
# warp_transform_path = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/bobby_head/transform_mni_to_bobby1InverseWarp.nii.gz"
warp_transform_path = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/ROI Registration/ccc7_24jan25_warp_inv.nii"

tract_names = ["M1_L", "M1_R", "PMd_L", "PMd_R", "preSMA_L", "preSMA_R", "SMA_L", "SMA_R"]
transformed_files = []

# Define distinct colormaps for each ROI (valid Matplotlib colormaps)
colormap_map = {
    "M1_L": "Reds",
    "M1_R": "Blues",
    "PMd_L": "Greens",
    "PMd_R": "YlOrBr",
    "preSMA_L": "Purples",
    "preSMA_R": "cool",
    "SMA_L": "magma",
    "SMA_R": "cividis",
}

# Step 1: Apply transformations if needed
for tract_name in tract_names:
    nii_path = f"/Users/arshsingh/Desktop/Research/Crest_CST_Work/Boyne Cortical Subregion Tracts/ROIs/{tract_name}.inDil2.nii"
    transformed_nii_path = f"{output_directory}/{tract_name}_warped_mni.nii.gz"

    if not os.path.exists(transformed_nii_path):
        apply_transform_cmd = (
            f"source ~/.bashrc && "
            f"antsApplyTransforms -d 3 "
            f"-i '{nii_path}' "
            f"-r '{fixed_image_path}' "
            f"-o '{transformed_nii_path}' "
            f"-t ['{transform_affine_path},1'] "
            f"-t ['{warp_affine_path}, 1'] "
            f"-t '{warp_transform_path}' "
            f"--interpolation GenericLabel"
        )

        try:
            print(apply_transform_cmd)
            subprocess.run(apply_transform_cmd, shell=True, check=True)
            print(f"Transformation successful for {tract_name}. File saved at: {transformed_nii_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error transforming {tract_name}: {e}")
            continue  # Skip if the transformation fails

    else:
        print(f"Transformed file already exists for {tract_name}, skipping transformation.")

    transformed_files.append((tract_name, transformed_nii_path))

# Step 2: Load brain mesh
msh_mesh = meshio.read(msh_path)
points = msh_mesh.points
meshio_to_vtk = {"triangle": 5, "tetra": 10, "hexahedron": 12, "quad": 9}
desired_layer = 1002

filtered_cells = []
for (cell_type, cell_data), layer_array in zip(msh_mesh.cells_dict.items(), msh_mesh.cell_data["gmsh:physical"]):
    if cell_type in meshio_to_vtk:
        mask = layer_array == desired_layer
        selected_cells = cell_data[mask]
        if len(selected_cells) > 0:
            num_cells = len(selected_cells)
            vtk_type = meshio_to_vtk[cell_type]
            cell_sizes = np.full((num_cells, 1), selected_cells.shape[1], dtype=np.int32)
            vtk_cells = np.hstack([cell_sizes, selected_cells]).flatten()
            filtered_cells.append((vtk_cells, np.full(num_cells, vtk_type, dtype=np.uint8)))

if filtered_cells:
    vtk_cells, vtk_types = zip(*filtered_cells)
    vtk_cells = np.concatenate(vtk_cells)
    vtk_types = np.concatenate(vtk_types)
    brain_mesh = pv.UnstructuredGrid(vtk_cells, vtk_types, points)
else:
    brain_mesh = None

# Step 3: Visualize all transformed tracts
plotter = pv.Plotter(border=False)
plotter.set_background("white")
plotter.add_mesh(brain_mesh, color="dimgray", opacity=0.08, label="Brain")

for tract_name, transformed_nii_path in transformed_files:
    try:
        nii_img = nib.load(transformed_nii_path)
    except FileNotFoundError:
        print(f"Error: Transformed file '{transformed_nii_path}' not found. Skipping visualization.")
        continue

    nii_data = nii_img.get_fdata()
    nii_affine = nii_img.affine
    spacing = np.abs([nii_affine[0, 0], nii_affine[1, 1], nii_affine[2, 2]])

    grid = pv.ImageData(dimensions=nii_data.shape)
    grid.spacing = spacing
    grid.origin = (nii_affine[0, 3], nii_affine[1, 3], nii_affine[2, 3])
    grid.point_data["Intensity"] = nii_data.ravel(order="F")

    plotter.add_volume(grid, scalars="Intensity", cmap=colormap_map[tract_name], opacity="sigmoid", show_scalar_bar=True)

plotter.camera_position = [(0, 800, 0), (0, 0, 0), (0, 0, 1)]
plotter.show()
