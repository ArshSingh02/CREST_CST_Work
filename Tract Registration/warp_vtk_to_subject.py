import numpy as np
import pandas as pd
import vtk
import subprocess
import os  # Import os for file deletion

# Change affine matrix and inverse warp to correct ones before running
def vtk_to_csv(directory, output_directory, tract_name):
    vtk_file = f"{directory}/{tract_name}.vtk"
    csv_file = f"{output_directory}/{tract_name}_inv_mni.csv"

    # Read VTK File
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    polydata = reader.GetOutput()

    # Read Coordinates in VTK File and Write into List
    coordinates = polydata.GetPoints()

    coordinate_list = []
    for i in range(coordinates.GetNumberOfPoints()):
        coordinate_list.append(coordinates.GetPoint(i))

    # Create Dataframe, invert X and Y, and write to CSV File
    df = pd.DataFrame(coordinate_list, columns=['x', 'y', 'z'])
    df['x'] = df['x'] * -1.0
    df['y'] = df['y'] * -1.0

    df.to_csv(csv_file, index=False)

    # Use affine matrix and inverse warp calculated from using antsRegistration on subject structural T1 mri and atlas T1 mri
    applyTransform = (
        "source ~/.bashrc && "
        + f"antsApplyTransformsToPoints -d 3 -i '{csv_file}' -o '{output_directory}/{tract_name}_warped_mni.csv' "
        + "-t ['/Users/arshsingh/Desktop/Research/Crest_CST_Work/bobby_head/transform_mni_to_bobby0GenericAffine.mat', 1] "
        + "-t '/Users/arshsingh/Desktop/Research/Crest_CST_Work/bobby_head/transform_mni_to_bobby1InverseWarp.nii.gz'"
    )

    try:
        subprocess.run(applyTransform, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing a command: {e}")


def csv_to_vtk(directory, output_directory, tract_name):
    original_vtk_file = f"{directory}/{tract_name}.vtk"
    transformed_csv_file = f"{output_directory}/{tract_name}_warped_mni.csv"
    output_vtk_file = f"{output_directory}/{tract_name}_warped_bobby_left.vtk"

    # Read the original VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(original_vtk_file)
    reader.Update()
    original_polydata = reader.GetOutput()

    # Read the transformed CSV file
    transformed_coords = pd.read_csv(transformed_csv_file)

    # Revert back to original axes direction
    transformed_coords['x'] = transformed_coords['x'] * -1
    transformed_coords['y'] = transformed_coords['y'] * -1

    new_points = vtk.vtkPoints()
    for _, row in transformed_coords.iterrows():
        new_points.InsertNextPoint(row['x'], row['y'], row['z'])

    # Create a new polydata object to store the new structure
    new_polydata = vtk.vtkPolyData()
    new_polydata.SetPoints(new_points)

    # Copy cells from the original polydata to the new one
    new_polydata.SetPolys(original_polydata.GetPolys())
    new_polydata.SetLines(original_polydata.GetLines())
    new_polydata.SetVerts(original_polydata.GetVerts())
    new_polydata.SetStrips(original_polydata.GetStrips())

    # Create vtkPolyData object and update with coordinates
    new_polydata.GetPointData().ShallowCopy(original_polydata.GetPointData())
    new_polydata.GetCellData().ShallowCopy(original_polydata.GetCellData())

    # Write the polydata to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_file)
    writer.SetInputData(new_polydata)
    writer.Write()

    # Remove the intermediate CSV file
    os.remove(transformed_csv_file)


if __name__ == "__main__":
    directory = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/Boyne Cortical Subregion Tracts/vtk File Format"
    output_directory = "/Users/arshsingh/Desktop/Research/Crest_CST_Work/Boyne Cortical Subregion Tracts/Warped Tracts"

    file_names = [
        "MNI_dsnFSL_IDmerge_CST_L_manual.prune_M1",
        "MNI_dsnFSL_IDmerge_CST_L_manual.prune_SMA",
        "MNI_dsnFSL_IDmerge_CST_L_manual.prune_preSMA",
        "MNI_dsnFSL_IDmerge_CST_L_manual.prune_PMd",
        "MNI_dsnFSL_IDmerge_CST_R_manual.prune_M1",
        "MNI_dsnFSL_IDmerge_CST_R_manual.prune_SMA",
        "MNI_dsnFSL_IDmerge_CST_R_manual.prune_preSMA",
        "MNI_dsnFSL_IDmerge_CST_R_manual.prune_PMd",
    ]

    for tract_name in file_names:
        vtk_to_csv(directory=directory, output_directory=output_directory, tract_name=tract_name)
        csv_to_vtk(directory=directory, output_directory=output_directory, tract_name=tract_name)

        # Delete the intermediate CSV files
        intermediate_csv_files = [
            f"{output_directory}/{tract_name}_inv_mni.csv",
            f"{output_directory}/{tract_name}_warped_mni.csv",
        ]
        for file in intermediate_csv_files:
            if os.path.exists(file):
                os.remove(file)
