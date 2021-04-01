"""
Main preprocessing functions.
Starting from many dicom files for each patient (many slices composing a 3D volume),
this functions apply cropping, reshaping, create labels from stl files and merge labels.
They also create corresponding Nifti files of cropped, reshaped, labeled and merged volumes.

"""

from __future__ import print_function
import trimesh
from stl import mesh
import xlrd
import nibabel as nib
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_erosion, binary_dilation
from skimage.morphology.selem import disk
from segmentation.preprocess.utils import *


def Volume_Crop():
    """
    Crop slices in x,y,z direction to minimize the amount of background voxels, to optimize memory.
    It saves at "filenameCT" .nii cropped files
    Returns:
    """
    # Set directories
    setDirVariables()

    # Options
    b_display = 0

    # Sets
    InDCMmSet = 'knee'
    InStlSet = 'stl'
    InSurfSet1 = 'ct_femur_'
    InSurfSet2 = 'ct_tibia_'
    InSurfSet3 = 'ct_fibula_'
    InSurfSet4 = 'ct_patella_'
    OutSurfSet = 'crop_knee'

    os.chdir(mainInputDataDirectoryNAS)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    fp.close()
    print('Patient no. {:04d}'.format(casePatient))

    # Read excel file to get patients' codes
    xlsName = os.path.join(mainInputDataDirectoryNAS, 'Case_statistics.xlsx')
    #name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]
    patientCode = study[casePatient - 1]


    ## Patient folder
    mainInputPatientDirectoryNAS = mainInputDataDirectoryNAS + '/OriginalData/' + patientCode

    # Load DICOM files
    mainInputDicomDirectory = mainInputPatientDirectoryNAS + '/dicom/'
    if os.path.isdir(mainInputDicomDirectory + '/ct/'):
        mainInputDicomDirectory = mainInputDicomDirectory + '/ct/' + InDCMmSetdicom + '/'
    else:
        mainInputDicomDirectory = mainInputDicomDirectory + InDCMmSetdicom + '/'
    VolumeCT = loadDicomVolume(mainInputDicomDirectory)
    VolumeCT.volumeData = np.flip(VolumeCT.volumeData, axis=2)  # flip on z axis to restore natural positions of bones

    # Read stl and display
    filename1 = InSurfSet1 + study[casePatient - 1] + '.stl'
    filename2 = InSurfSet2 + study[casePatient - 1] + '.stl'
    filename3 = InSurfSet3 + study[casePatient - 1] + '.stl'
    filename4 = InSurfSet4 + study[casePatient - 1] + '.stl'

    mainInputStlDirectory = mainInputPatientDirectoryNAS + '/' + InStlSet + '/'
    os.chdir(mainInputStlDirectory)

    meshes = []
    # Femur
    if os.path.isfile(filename1):
        meshes.append(mesh.Mesh.from_file(filename1))
    # Tibia
    if os.path.isfile(filename2):
        meshes.append(mesh.Mesh.from_file(filename2))
    # Fibula
    if os.path.isfile(filename3):
        meshes.append(mesh.Mesh.from_file(filename3))
    # Patella
    if os.path.isfile(filename4):
        meshes.append(mesh.Mesh.from_file(filename4))

    if len(meshes) == 2:
        new_mesh = mesh.Mesh(np.concatenate([meshes[0].data, meshes[1].data]))
    elif len(meshes) == 3:
        new_mesh = mesh.Mesh(np.concatenate([meshes[0].data, meshes[1].data]))
        new_mesh = mesh.Mesh(np.concatenate([new_mesh.data, meshes[2].data]))
    elif len(meshes) == 4:
        new_mesh = mesh.Mesh(np.concatenate([meshes[0].data, meshes[1].data]))
        new_mesh = mesh.Mesh(np.concatenate([new_mesh.data, meshes[2].data]))
        new_mesh = mesh.Mesh(np.concatenate([new_mesh.data, meshes[3].data]))
    elif len(meshes) == 0:
        print('*** Stl file not found ***')
        return
    else:
        new_mesh = meshes[0]

    CropVolumeCT = VolumeCT
    deltaNumberofSlices = 2

    # CROP THE VOLUME IN ALL THREE DIRECTIONS Z, Y, X

    indexCoord = 0  # Sagittal slicing direction (X direction) (columns)
    indexVolume = 1
    minValue = new_mesh.min_[indexCoord]
    maxValue = new_mesh.max_[indexCoord]
    CropVolumeCT = cropVolume(CropVolumeCT, deltaNumberofSlices, minValue, maxValue, indexCoord, indexVolume)

    indexCoord = 1  # frontal slicing direction (Y direction) (rows)
    indexVolume = 0
    minValue = new_mesh.min_[indexCoord]
    maxValue = new_mesh.max_[indexCoord]
    CropVolumeCT = cropVolume(CropVolumeCT, deltaNumberofSlices, minValue, maxValue, indexCoord, indexVolume)

    indexCoord = 2  # axial slicing direction (Z direction) (slices)
    indexVolume = indexCoord
    minValue = new_mesh.min_[indexCoord]
    maxValue = new_mesh.max_[indexCoord]
    CropVolumeCT = cropVolume(CropVolumeCT, deltaNumberofSlices, minValue, maxValue, indexCoord, indexVolume)

    volumeData = np.transpose(CropVolumeCT.volumeData, [1, 0, 2])
    volumeOffset = CropVolumeCT.volumeOffset
    affine = np.eye(4)
    niiCropVolumeCT = nib.Nifti1Image(volumeData, affine)
    niiCropVolumeCT.header.set_slope_inter(CropVolumeCT.rescaleSlope, CropVolumeCT.rescaleIntercept)
    #niiCropVolumeCT.header['scl_slope'] = int(CropVolumeCT.rescaleSlope)
    #niiCropVolumeCT.header['scl_inter'] = int(CropVolumeCT.rescaleIntercept)
    niiCropVolumeCT.header.set_qform(affine, 1)
    niiCropVolumeCT.header.set_zooms(CropVolumeCT.voxelSize)
    niiCropVolumeCT.header['qoffset_x'] = volumeOffset[0]
    niiCropVolumeCT.header['qoffset_y'] = volumeOffset[1]
    niiCropVolumeCT.header['qoffset_z'] = volumeOffset[2]
    # *** OFFSET SETTING MISSING ***

    # Save file to nii format
    patientFolder = 'Patient{:03d}'.format(casePatient)
    outputPatientDirectory = mainOutputDataDirectoryLoc + '/' + patientFolder + '/'
    mainOutputPatientDirectory = outputPatientDirectory
    os.chdir(mainOutputPatientDirectory)
    filenameCT = 'volumeCT_' + OutSurfSet + '_{:04d}_py.nii'.format(casePatient)
    nib.nifti1.save(niiCropVolumeCT, filenameCT)


def Volume_Reshape():
    """
    Reshapes all .nii cropped files to the same dimensions along x,y and z.
    It saves at "filenameCT"" the resampled .nii volumes.
    Returns:

    """

    setDirVariables()

    # Options
    b_display = 0
    # Resolution
    resolution = [1, 1, 1]
    # Dimension
    Dimension = [192, 192, 192]

    # Sets
    InDCMmSet = 'knee'
    InNiiSet = 'crop_knee'
    OutSurfSet = 'reshape_knee_' + str(Dimension[0]) + '_' + str(Dimension[1]) + '_' + str(Dimension[2])

    os.chdir(mainInputDataDirectoryLoc)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    # casePatient = 3
    fp.close()
    print('Patient no. {:04d}'.format(casePatient))

    # Read excel file to get patients' codes
    xlsName = os.path.join(mainInputDataDirectoryLoc, 'Case_statistics.xlsx')
    # name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]
    patientCode = study[casePatient - 1]

    ## Read volume nii
    patientDirectory = 'Patient{:04d}'.format(casePatient)
    mainInputPatientDirectoryLoc = mainInputDataDirectoryLoc + '/preprocessedData/' + patientDirectory + '/'
    mainInputPatientDirectoryNAS = mainInputDataDirectoryNAS + '/OriginalData/' + patientCode
    mainInputDicomDirectory = mainInputPatientDirectoryNAS + '/dicom/'
    if os.path.isdir(mainInputDicomDirectory + '/ct/'):
        mainInputDicomDirectory = mainInputDicomDirectory + '/ct/' + InDCMmSetdicom + '/'
    else:
        mainInputDicomDirectory = mainInputDicomDirectory + InDCMmSetdicom + '/'

    os.chdir(mainInputPatientDirectoryLoc)

    # Read NII
    niiFilename = 'volumeCT_' + InNiiSet + '_{:04d}.nii'.format(casePatient)
    VolumeCT = loadNiiVolume(niiFilename, mainInputDicomDirectory)
    # Normalize
    VolumeCT = normVolumeScan(VolumeCT)

    os.chdir(mainCodeDirectory)

    # Get data
    volumeDim = VolumeCT.volumeDim
    volumeVoxSize = VolumeCT.voxelSize
    dimX = (volumeDim[0] - 1) * volumeVoxSize[0]
    dimY = (volumeDim[1] - 1) * volumeVoxSize[1]
    dimZ = (volumeDim[2] - 1) * volumeVoxSize[2]
    nVolumeDim = [round(dimX / resolution[0] + 1), round(dimY / resolution[1] + 1), round(dimZ / resolution[2] + 1)]

    # Resample to new dimension
    VolumeResampled = resampleVolumeScan(VolumeCT, Dimension)
    # Normalize between 0 and 1
    VolumeResampled.volumeData = VolumeResampled.volumeData - VolumeResampled.volumeData.min()
    VolumeResampled.volumeData = VolumeResampled.volumeData/VolumeResampled.volumeData.max()
    volumeOffset = VolumeResampled.volumeOffset

    affine = np.eye(4)
    niiResampledVolumeCT = nib.Nifti1Image(VolumeResampled.volumeData, affine)
    niiResampledVolumeCT.header.set_slope_inter(VolumeResampled.rescaleSlope, VolumeResampled.rescaleIntercept)
    niiResampledVolumeCT.header.set_qform(affine, 1)
    niiResampledVolumeCT.header.set_zooms(VolumeResampled.voxelSize)
    niiResampledVolumeCT.header['qoffset_x'] = volumeOffset[0]
    niiResampledVolumeCT.header['qoffset_y'] = volumeOffset[1]
    niiResampledVolumeCT.header['qoffset_z'] = volumeOffset[2]

    os.chdir(mainInputPatientDirectoryLoc)

    # Save nii
    filenameCT = 'volumeCT_' + OutSurfSet + '_{:04d}_py.nii'.format(casePatient)
    nib.nifti1.save(niiResampledVolumeCT, filenameCT)

    os.chdir(mainCodeDirectory)


def Volume_Label():
    """
    Creates binary labels from .stl files, that are sliced at height locations
    given by the position of the slices in the corresponding volume.nii file.
    It Saves at "filenamelabel" the binary labels of the selected anatomy.
    Returns:

    """
    setDirVariables()

    ## Options
    # b_display = 0
    # compare_Matlab_Python = 0

    # Output size

    outSize = [128, 128, 128]
    templateSize = str(outSize[0]) + '_' + str(outSize[1]) + '_' + str(outSize[2])

    # Sets
    InStlSet = 'stl'
    InDCMmSet = 'reshape_knee_'
    InDCMmSetdicom = 'knee'

    fileTemplate = InDCMmSet + templateSize
    anatomy = 'tibia'  # 'femur', 'patella', 'fibula'
    position = 'prox'  # 'dist', '', ''
    InSurfSet = 'ct_' + anatomy + '_'
    outSet = position + anatomy

    os.chdir(mainInputDataDirectoryLoc)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    fp.close()
    print('Patient no. {}'.format(casePatient))

    # Read excel file to get patients' codes
    xlsName = os.path.join(mainInputDataDirectoryLoc, 'Case_statistics.xlsx')
    # name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]
    patientCode = study[casePatient - 1]

    ## Read volume nii
    mainPatientDirectory = 'Patient{:04d}'.format(casePatient)
    mainInputPatientDirectoryLoc = mainInputDataDirectoryLoc + '/preprocessedData/' + mainPatientDirectory + '/'
    mainInputPatientDirectoryNAS = mainInputDataDirectoryNAS + '/OriginalData/' + patientCode
    mainInputDicomDirectory = mainInputPatientDirectoryNAS + '/dicom/'
    if os.path.isdir(mainInputDicomDirectory + '/ct/'):
        mainInputDicomDirectory = mainInputDicomDirectory + '/ct/' + InDCMmSetdicom + '/'
    else:
        mainInputDicomDirectory = mainInputDicomDirectory + InDCMmSetdicom + '/'

    os.chdir(mainInputPatientDirectoryLoc)

    niiFilename = 'volumeCT_' + fileTemplate + '_{:04d}.nii'.format(casePatient)
    VolumeCT = loadNiiVolume(niiFilename, mainInputDicomDirectory)
    ## Normalize
    VolumeCT = normVolumeScan(VolumeCT)

    # Read stl and display
    mainInputStlDirectory = mainInputPatientDirectoryNAS + '/' + InStlSet + '/'
    os.chdir(mainInputStlDirectory)
    filename = InSurfSet + study[casePatient - 1] + '.stl'
    my_mesh1 = trimesh.load(filename)

    os.chdir(mainCodeDirectory)

    # Build  binary volume of the reference surface corresponding to the CT volume
    VolumeSurf = VolumeCT
    VolumeSurf.volumeData = np.zeros(VolumeSurf.volumeData.shape, dtype=int)

    heights = []
    for i in range(VolumeSurf.volumeDim[2]):
        heights.append(float(VolumeSurf.volumeOffset[2]) + i * VolumeSurf.voxelSize[2])

    contours = mesh2vol(my_mesh1, heights, VolumeSurf.volumeOffset, VolumeSurf.voxelSize, VolumeSurf.volumeDim)
    indicesX = []
    indicesY = []

    for ip in range(VolumeSurf.volumeDim[2]):

        if contours[ip].shape[0] != 0:

            val = contours[ip][:, 0] - VolumeSurf.volumeOffset[0]
            val = val / (VolumeSurf.volumeDim[0] * VolumeSurf.voxelSize[0])
            val = np.round(val * VolumeSurf.volumeDim[0], 0)
            valX = val.astype(int)

            val = contours[ip][:, 1] - VolumeSurf.volumeOffset[1]
            val = val / (VolumeSurf.volumeDim[1] * VolumeSurf.voxelSize[1])
            val = np.round(val * VolumeSurf.volumeDim[0], 0)
            valY = val.astype(int)

            val_index = np.zeros((valX.shape[0], 2))
            val_index[:, 0] = valX
            val_index[:, 1] = valY
            val_index = np.unique(val_index, axis=0).astype(int)

            indicesX.append(val_index[:, 0])
            indicesY.append(val_index[:, 1])

            for i, j in zip(valY, valX):
                VolumeSurf.volumeData[i - 1, j - 1, ip] = 1
        else:
            indicesX.append([])
            indicesY.append([])

    VolumeSurfLabeled = VolumeSurf
    counter1 = 0
    counter2 = 0
    # fill in the image each contour
    for ip in range(VolumeSurfLabeled.volumeDim[2]):

        if contours[ip].shape[0] != 0:

            non_zero_start = np.count_nonzero(VolumeSurf.volumeData[:, :, ip])

            ######## REGION FILL
            binaryImage = binary_fill_holes(VolumeSurf.volumeData[:, :, ip])
            binaryImage = binaryImage > 1 / 255
            ######### CLOSING
            kernel = np.ones((5, 5), np.uint8)
            binaryImage = binary_closing(binaryImage, kernel)
            ######### FILL HOLES AGAIN
            binaryImage = binary_fill_holes(binaryImage)

            non_zero_end = np.count_nonzero(binaryImage)

            ######### ALTERNATIVE PROCESSING FOR NON CLOSED CONTOURS
            if non_zero_end < non_zero_start * 4:
                strel = disk(2)
                binaryImage = binary_dilation(VolumeSurf.volumeData[:, :, ip], strel)
                binaryImage = binary_dilation(binaryImage, strel)
                binaryImage = binary_fill_holes(binaryImage)
                binaryImage = binary_erosion(binaryImage, strel)
                binaryImage = binary_erosion(binaryImage, strel)
                counter1 = counter1 + 1

                non_zero_end2 = np.count_nonzero(binaryImage)

                ######### ALTERNATIVE PROCESSING FOR STILL-NON-CLOSED CONTOURS
                if non_zero_end2 < non_zero_start * 4:
                    strel = disk(3)
                    binaryImage = binary_dilation(VolumeSurf.volumeData[:, :, ip], strel)
                    binaryImage = binary_dilation(binaryImage, strel)
                    binaryImage = binary_fill_holes(binaryImage)
                    binaryImage = binary_erosion(binaryImage, strel)
                    binaryImage = binary_erosion(binaryImage, strel)
                    counter2 = counter2 + 1

            VolumeSurfLabeled.volumeData[:, :, ip] = binaryImage

            dMin = VolumeSurfLabeled.volumeData.min()
            D = VolumeSurfLabeled.volumeData + abs(dMin)
            D = D / D.max() * 255

    print('Alternative processing no 1: {} \n Alternative proessing no 2: {}'.format(counter1, counter2))

    ###### PLOT AND SCROLL ACROSS SLICES
    #if b_display == 1:
    #    fig, ax = plt.subplots(1, 1)
    #    tracker = IndexTracker(ax, D)
    #    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    #    plt.show()
    #
    #if compare_Matlab_Python == 1:
    #    name_dir = outSet + 'Label'
    #    mainLabelDirectory = os.path.join(mainPatientDirectory, '{}'.format(name_dir))
    #    os.chdir(mainLabelDirectory)
    #    mean_dice = dice_coeff(VolumeSurfLabeled.volumeData, outSet)
    #    print(mean_dice)

    # Make nii file label
    volumeData = VolumeSurfLabeled.volumeData.astype(np.short)
    volumeData = np.transpose(volumeData, [1, 0, 2])
    voxelSize = VolumeSurfLabeled.voxelSize
    volumeOffset = VolumeSurfLabeled.volumeOffset

    affine = np.eye(4)
    niiVolumeLabel = nib.Nifti1Image(volumeData, affine)
    niiVolumeLabel.header.set_slope_inter(VolumeSurfLabeled.rescaleSlope, VolumeSurfLabeled.rescaleIntercept)
    niiVolumeLabel.header.set_qform(affine, 1)
    niiVolumeLabel.header.set_zooms(voxelSize)
    niiVolumeLabel.header['qoffset_x'] = volumeOffset[0]
    niiVolumeLabel.header['qoffset_y'] = volumeOffset[1]
    niiVolumeLabel.header['qoffset_z'] = volumeOffset[2]

    os.chdir(mainInputPatientDirectoryLoc)

    # Save nii
    filenameLabel = 'volumeLabel_' + outSet + '_' + templateSize + '_{:04d}_py.nii'.format(casePatient)
    nib.nifti1.save(niiVolumeLabel, filenameLabel)

    os.chdir(mainCodeDirectory)


def Merge_Labels():

    """
    Merges label files of theh desired anatomies
    It saves at ""pathname" the .nii file that stores the merged binary labels.
    Returns:
    """
    setDirVariables()

    # Options
    b_display = 0
    # Output size

    outSize = [128, 128, 128]
    templateSize = str(outSize[0]) + '_' + str(outSize[1]) + '_' + str(outSize[2])

    # Sets
    InDCMmSetdicom = 'knee'
    anatomy1 = 'femur'
    InSurfSet1 = 'dist' + anatomy1 + '_' + templateSize
    anatomy2 = 'tibia'
    InSurfSet2 = 'prox' + anatomy2 + '_' + templateSize
    anatomy3 = 'patella'
    InSurfSet3 = anatomy3 + '_' + templateSize
    anatomy4 = 'fibula'
    InSurfSet4 = anatomy4 + '_' + templateSize

    templateOut = 'reshape_knee_' + templateSize

    os.chdir(mainInputDataDirectoryLoc)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    fp.close()
    print('Patient no. {:04d}'.format(casePatient))

    # Read excel file to get patients' codes
    xlsName = os.path.join(mainInputDataDirectoryLoc, 'Case_statistics.xlsx')
    # name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]
    patientCode = study[casePatient - 1]

    mainPatientDirectory = 'Patient{:03d}'.format(casePatient)
    mainInputPatientDirectoryLoc = mainInputDataDirectoryLoc + '/preprocessedData/' + mainPatientDirectory + '/'
    mainInputPatientDirectoryNAS = mainInputDataDirectoryNAS + '/OriginalData/' + patientCode
    mainInputDicomDirectory = mainInputPatientDirectoryNAS + '/dicom/'
    if os.path.isdir(mainInputDicomDirectory + '/ct/'):
        mainInputDicomDirectory = mainInputDicomDirectory + '/ct/' + InDCMmSetdicom + '/'
    else:
        mainInputDicomDirectory = mainInputDicomDirectory + InDCMmSetdicom + '/'

    os.chdir(mainInputPatientDirectoryLoc)

    # Read label volume1
    pathName1 = 'volumeLabel_' + InSurfSet1 + '_{:04d}.nii'.format(casePatient)
    # pathName1 = mainPatientDirectory + pathName1
    VolumeLabel1 = loadNiiVolume(pathName1, mainInputDicomDirectory)

    # Read label volume2
    pathName2 = 'volumeLabel_' + InSurfSet2 + '_{:04d}.nii'.format(casePatient)
    # pathName2 = mainPatientDirectory + pathName2
    VolumeLabel2 = loadNiiVolume(pathName2, mainInputDicomDirectory)

    # Read label volume3
    pathName3 = 'volumeLabel_' + InSurfSet3 + '_{:04d}.nii'.format(casePatient)
    # pathName3 = mainPatientDirectory + pathName1
    if os.path.exists(pathName3):
        VolumeLabel3 = loadNiiVolume(pathName3, mainInputDicomDirectory)

    # Read label volume4
    pathName4 = 'volumeLabel_' + InSurfSet4 + '_{:04d}.nii'.format(casePatient)
    # pathName4 = mainPatientDirectory + pathName1
    if os.path.exists(pathName4):
        VolumeLabel4 = loadNiiVolume(pathName4, mainInputDicomDirectory)

    ## MERGE WANTED LABELS
    # Increase index of the label
    indexLabel = np.where(VolumeLabel2.volumeData == 1)
    VolumeLabel2.volumeData[indexLabel] = VolumeLabel2.volumeData[indexLabel] + 1

    VolumeLabelJoin = VolumeLabel1
    VolumeLabelJoin.volumeData = VolumeLabelJoin.volumeData + VolumeLabel2.volumeData

    # Make nii Label file
    volumeData = VolumeLabelJoin.volumeData.astype(np.short)
    voxelSize = VolumeLabelJoin.voxelSize
    volumeOffset = VolumeLabelJoin.volumeOffset

    affine = np.eye(4)
    niiVolumeLabelJoin = nib.Nifti1Image(volumeData, affine)
    niiVolumeLabelJoin.header.set_slope_inter(VolumeLabelJoin.rescaleSlope, VolumeLabelJoin.rescaleIntercept)
    niiVolumeLabelJoin.header.set_qform(affine, 1)
    niiVolumeLabelJoin.header.set_zooms(voxelSize)
    niiVolumeLabelJoin.header['qoffset_x'] = volumeOffset[0]
    niiVolumeLabelJoin.header['qoffset_y'] = volumeOffset[1]
    niiVolumeLabelJoin.header['qoffset_z'] = volumeOffset[2]

    # Save joined labels
    filenameOut = 'volumeLabel_' + templateOut + '_{:04d}.nii'.format(casePatient)
    pathName = mainInputPatientDirectoryLoc + filenameOut
    nib.nifti1.save(niiVolumeLabelJoin, pathName)

    os.chdir(mainCodeDirectory)
