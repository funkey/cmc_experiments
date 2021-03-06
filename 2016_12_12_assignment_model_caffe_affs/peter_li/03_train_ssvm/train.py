import os
from subprocess import call, Popen, PIPE
import sys
sys.path.append('..')
from add_rf_feature import add_rf_feature
import datetime

def tee(cmd, log_file = None):

    full_cmd = ""
    for i in cmd:
        full_cmd += i + " "
    print("running: " + full_cmd)

    if log_file is None:

        call(cmd)

    else:

        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)

        with open(log_file, "w") as f:
            while proc.poll() is None:
                line = proc.stdout.readline()
                while line:
                    print (line.strip())
                    f.write(line)
                    line = proc.stdout.readline()
                line = proc.stderr.readline()
                while line:
                    print (line.strip())
                    f.write(line)
                    line = proc.stderr.readline()

if __name__ == "__main__":

    if not os.path.exists("tif"):
        os.mkdir("tif")
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("hdf"):
        os.mkdir("hdf")

    sample = 'A'

    ts = datetime.datetime.now()
    tee([
        "cmc_create_project",
        "--mergeHistoryWithScores",
        "--maxMerges=8",
        "--2dSupervoxels",
        "--resX=4",
        "--resY=4",
        "--resZ=40",
        "--cragType=empty",
        "--maxZLinkBoundingBoxDistance=400",
        "--supervoxels=../01_data_preparation/train_ssvm/fragments" + sample,
        "--mergeHistory=../01_data_preparation/train_ssvm/mergetrees" + sample,
        "--groundTruth=../01_data_preparation/train_ssvm/groundtruth" + sample,
        "--intensities=../01_data_preparation/train_ssvm/raw" + sample,
        "--boundaries=../01_data_preparation/train_ssvm/membrane" + sample,
        "--xAffinities=../01_data_preparation/train_ssvm/affinities" + sample + "/aff_x",
        "--yAffinities=../01_data_preparation/train_ssvm/affinities" + sample + "/aff_y",
        "--zAffinities=../01_data_preparation/train_ssvm/affinities" + sample + "/aff_z",
        "--project=hdf/train_ssvm.hdf",
        "--importTrainingResult=../02_train_rf/hdf/train_rf.hdf"
    ], "log/create_project.log")

    # extract features
    tee([
        "cmc_extract_features",
        "--log-level=debug",
        "--project=hdf/train_ssvm.hdf",
        "--statisticsFeatures=true",
        "--topologicalFeatures=true",
        "--assignmentFeatures=true",
        "--addPairwiseProducts=true",
        "--normalize=true",
        "--minMaxFromProject=true",
        "--dumpFeatureNames=true"
    ], "log/extract_features.log")

    add_rf_feature("../02_train_rf/hdf/train_rf.hdf", "hdf/train_ssvm.hdf")

    # train ssvm
    tee([
        "cmc_train",
        "--log-level=debug",
        "--project=hdf/train_ssvm.hdf",
        "--assignmentSolver",
        "--exportBestEffort=tif/best-effort"
    ], "log/best_effort.log")

    tf = datetime.datetime.now()
    print ("time: ")
    print (tf-ts)
