import os
import re
import cv2
import csv
import netCDF4
import argparse
import numpy as np
import matplotlib.pyplot as plt
from .utils.predict_helpers import calculate_mpf, filter_and_calculate_mpf, predict_image_autosam, predict_image_smp, crop_center_square, label_to_pixelvalue, label_to_pixelvalue_with_uncertainty, predict_image_smp, extract_features


# TO-DO: clean up arguments and prevent errors


parser = argparse.ArgumentParser(
    description="Uses trained model to predict and store surface masks from netCDF file containing TIR images from a single helicopter flight. Optional calculation of melt pond fraction (MPF)."
)

parser.add_argument(
    "--pref",
    type=str,
    default="001",
    help="Identifier for the current prediction. Will be used as foldername to store results.",
)

parser.add_argument(
    "--arch",
    type=str,
    default="UnetPlusPlus",
    choices=["Unet", "AutoSAM", "UnetPlusPlus", "Linknet", "MAnet"],
    help="Model Architecture to use.",
)

parser.add_argument(
    "--data",
    default="data/prediction/temperatures/IRdata_ATWAICE_processed_220718_142920.nc",
    type=str,
    help="Either: 1) Filename of netCDF data file. For this, data must be stored in 'data/prediction/raw'. Or: 2) Absolute path to netCDF data file. Then data must not be copied in advance.",
)
parser.add_argument(
    "--weights_path",
    default="final_checkpoints/UnetPlusPlus/inference_UnetPlusPlus_aid.pth",
    type=str,
    help="Path to model weights that should be used. Must contain the model architecture as second-to-last part of path (should be per default).",
)
parser.add_argument(
    "--preprocessed_path",
    default="data/prediction/preprocessed",
    type=str,
    help="Path to folder that should store the preprocessed images.",
)
parser.add_argument(
    "--predicted_path",
    default="data/prediction/predicted",
    type=str,
    help="Path to folder that should store the predicted image masks.",
)
parser.add_argument(
    "--autosam_size",
    default="vit_b",
    type=str,
    help="Model type that should be used. Must be the same as in 'weights_path'.",
)
parser.add_argument(
    "--skip_mpf",
    default=False,
    action="store_true",
    help="Skips the calculation of the melt pond fraction for the predicted flight.",
)
parser.add_argument(
    "--skip_preprocessing",
    default=False,
    action="store_true",
    help="Skips preprocessing. Can be used to directly perform mpf calculation. In that case, 'predicted_path' must contain predicted images.",
)
parser.add_argument(
    "--skip_prediction",
    default=False,
    action="store_true",
    help="Skips prediction process. Can be used to directly perform mpf calculation. In that case, 'predicted_path' must contain predicted images.",
)
parser.add_argument(
    "--skip_convert_to_grayscale",
    default=False,
    action="store_true",
    help="Converts predicted images to grayscale for visualization and stores in 'data/prediction/predicted/[pref]/grayscale'.",
)
parser.add_argument(
    "--normalize",
    default=False,
    action="store_true",
    help="Normalize images before prediction. Should be used if model was trained with normalized images.",
)
parser.add_argument(
    "--no_finetune",
    default=False,
    action="store_true",
    help="Should be used if model was trained without finetuning.",
)
parser.add_argument("--model", type=str, default="autosam")
parser.add_argument(
    "--val_predict",
    default=False,
    action="store_true",
    help="to be activated when predicting validation imgs",
)
parser.add_argument(
    "--create_output_nc",
    default=False,
    action="store_true",
    help="Create output netCDF file with predicted masks and output fractions.",
)
parser.add_argument(
    "--plot_with_uncertainty",
    default=False,
    action="store_true",
    help="Plot masks with uncertainty.",
)
parser.add_argument(
    "--get_features",
    default=False,
    action="store_true",
)


def main():
    args = parser.parse_args()

    # add prefix to storage paths and create folder
    args.predicted_path = os.path.join(args.predicted_path, args.pref)
    os.makedirs(args.predicted_path, exist_ok=True)
    output_path = os.path.join("runs", args.pref)
    os.makedirs(output_path, exist_ok=True)

    if args.data == "none":
        print("Data is none. Must be specified.")

    id = args.data.split("/")[-1]

    # extract date of flight used
    match = re.search(r"(\d{6})_(\d{6})", args.data)

    if match:
        date_part = match.group(1)

        # formatting the date
        formatted_date = f"20{date_part[:2]}-{date_part[2:4]}-{date_part[4:]}"
        print(f"The date in the filename is: {formatted_date}")

        if not args.val_predict:
            args.preprocessed_path = os.path.join(args.preprocessed_path, id)

        os.makedirs(args.preprocessed_path, exist_ok=True)

    else:
        print("Date not found in the filename.")

    # extract model architecture from weights_path
    model_arch = args.weights_path.split("/")[2]
    print("Model architecture used: ".format(model_arch))

    if not args.skip_preprocessing:
        # load data and store as images
        # use whole path when abs path is given, else use data from 'data/prediction/raw'
        if "/" in args.data:
            ds = netCDF4.Dataset(args.data)
        else:
            ds = netCDF4.Dataset(
                os.path.join("data/prediction/raw", id, args.data)
            )
        imgs = ds.variables["Ts"][:]

        tmp = []

        for im in imgs:
            im = crop_center_square(im)
            tmp.append(im)

        imgs = tmp

        print("Start extracting images...")

        new_idx = 0

        for idx, img in enumerate(imgs):
            #if(idx % 4 == 0):
            # convert temperature values to grayscale images
            plt.imsave(
                os.path.join(args.preprocessed_path, "{}.png".format(new_idx)),
                img,
                cmap="gray",
            )
            new_idx += 1

    if not args.skip_prediction:
        print("Start predicting images...")
        masks = np.zeros((len(os.listdir(args.preprocessed_path)), 480, 480)).astype(np.uint8)
        probabilities = np.zeros((len(os.listdir(args.preprocessed_path)), 3, 480, 480)).astype(np.float32)

        # extract surface masks from images
        for idx, file in enumerate(os.listdir(args.preprocessed_path)):
            print(masks.dtype)
            os.makedirs(os.path.join(args.predicted_path, "raw/"), exist_ok=True)
            id = file.split(".")[0]

            if file.endswith(".png"):
                img = cv2.imread(os.path.join(args.preprocessed_path, file), 0)

                if args.arch == "AutoSAM":
                    masks[idx], probabilities[idx] = predict_image_autosam(
                        img=img,
                        im_size=480,
                        weights=args.weights_path,
                        pretraining="sa-1b",
                        autosam_size=args.autosam_size,
                        save_path=os.path.join(
                            args.predicted_path, "raw/{}.png".format(id)
                        ),
                        normalize=args.normalize,
                        no_finetune=args.no_finetune,
                    )
                else:
                    masks[idx], probabilities[idx] = predict_image_smp(
                        arch=args.arch,
                        img=img,
                        im_size=480,
                        weights=args.weights_path,
                        pretraining="aid",
                        save_path=os.path.join(
                            args.predicted_path, "raw/{}.png".format(id)
                        ),
                        normalize=args.normalize,
                        no_finetune=args.no_finetune,
                    )
        os.makedirs(os.path.join(args.predicted_path, "np/"), exist_ok=True)
        np.save(os.path.join(args.predicted_path, "np/", "masks.npy"), np.array(masks))
        np.save(os.path.join(args.predicted_path, "np/", "probabilities.npy"), np.array(probabilities))

    if args.get_features:
        assert args.arch != "AutoSAM", "Feature extraction not implemented for AutoSAM."
        print("Start extracting features...")
        features = np.zeros((len(os.listdir(args.preprocessed_path)), 512, 15, 15)).astype(np.uint8)

        # extract surface masks from images
        for idx, file in enumerate(os.listdir(args.preprocessed_path)):
            print(features.dtype)
            os.makedirs(os.path.join(args.predicted_path, "raw/"), exist_ok=True)
            id = file.split(".")[0]

            if file.endswith(".png"):
                img = cv2.imread(os.path.join(args.preprocessed_path, file), 0)

                element = extract_features(
                    arch=args.arch,
                    img=img,
                    im_size=480,
                    weights=args.weights_path,
                    pretraining="aid",
                    save_path=os.path.join(
                        args.predicted_path, "raw/{}.png".format(id)
                    ),
                    normalize=args.normalize,
                    no_finetune=args.no_finetune,
                )[-1].detach().cpu().numpy()  # get the deepest features
                features[idx] = element
        os.makedirs(os.path.join(args.predicted_path, "np/"), exist_ok=True)
        np.save(os.path.join(args.predicted_path, "np/", "features.npy"), np.array(features))

    # optionally convert to grayscale images for visibility
    if not args.skip_convert_to_grayscale:
        os.makedirs(os.path.join(args.predicted_path, "grayscale/"), exist_ok=True)

        for idx, file in enumerate(
            os.listdir(os.path.join(args.predicted_path, "raw/"))
        ):
            id = file.split(".")[0]
            im = label_to_pixelvalue(
                cv2.imread(os.path.join(args.predicted_path, "raw/", file))
            )
            cv2.imwrite(
                os.path.join(args.predicted_path, "grayscale/{}.png".format(id)),
                im,
            )

    if args.plot_with_uncertainty:
        threshold = 0.9

        mask_dir = os.path.join(args.predicted_path, "np/", "masks.npy")
        probabilities_dir = os.path.join(args.predicted_path, "np/", "probabilities.npy")

        os.makedirs(os.path.join(args.predicted_path, "uncertainty/"), exist_ok=True)
    
        masks = np.load(mask_dir)
        probabilities = np.load(probabilities_dir)
        probabilities = probabilities.max(axis=1)
        print(probabilities.mean())

        for idx, im in enumerate(masks):
            prob_mask = np.where(probabilities[idx] < threshold)
            im[prob_mask] = 3
            im = label_to_pixelvalue_with_uncertainty(im)
            cv2.imwrite(
                os.path.join(args.predicted_path, "uncertainty/{}.png".format(idx)),
                im,
            )

    if args.create_output_nc:
        print("Creating output netCDF file...")
        mask_dir = os.path.join(args.predicted_path, "np/", "masks.npy")
        probabilities_dir = os.path.join(args.predicted_path, "np/", "probabilities.npy")
        masks = np.load(mask_dir)
        probabilities = np.load(probabilities_dir)
        output_nc_dir = os.path.join("data/prediction/output_nc/")
        os.makedirs(output_nc_dir, exist_ok=True)

        import netCDF4 as nc

        toexclude = ['Ts', 'grad']
        crop = ['yd', 'xd']

        with nc.Dataset(args.data) as src, nc.Dataset('data/prediction/output_nc/classified_{}'.format(id), 'w') as dst:           
            # copy global attributes all at once via dictionary
            dst.setncatts(src.__dict__)
            # copy dimensions
            for name, dimension in src.dimensions.items():
                if name == 'y':
                    dst.createDimension(
                        name, (480 if not dimension.isunlimited() else None))
                else:
                    dst.createDimension(
                        name, (len(dimension) if not dimension.isunlimited() else None))
            # copy all file data except for the excluded
            for name, variable in src.variables.items():
                if name not in toexclude:
                    to_np = src.variables[name][:]
                    if name in crop:
                        x = dst.createVariable(name, variable.datatype, ('t', 'y', 'x'))
                        for i in range(len(to_np)):
                            cropped = crop_center_square(to_np[i]).reshape(1,480,480)
                            dst[name][i] = cropped
                    else:
                        x = dst.createVariable(name, variable.datatype, variable.dimensions)
                        dst[name][:] = src[name][:]
                    # copy variable attributes all at once via dictionary
                    dst[name].setncatts(src[name].__dict__)
            # create mask variable
            mask = dst.createVariable('mask', 'float32', ('t', 'y', 'x'))
            mask.units = '0: melt pond, 1: sea ice, 2: open water'
            mask.grid_mapping = 'crs'
            mask[:] = masks

            # create fraction variables
            fractions = calculate_mpf(mask_dir)
            _, mean_probabilities = filter_and_calculate_mpf(
                mask_dir, probabilities_dir, threshold=0.9
            )

            mpf = dst.createVariable("mpf", "f4", ('t'))
            ocf = dst.createVariable("ocf", "f4", ('t'))
            sif = dst.createVariable("sif", "f4", ('t'))
            mpf.units = 'Melt Pond Fraction'
            ocf.units = 'Open Water Fraction'
            sif.units = 'Sea Ice Fraction'
            mpf[:] = fractions[:, 0]
            ocf[:] = fractions[:, 1]
            sif[:] = fractions[:, 2]

            # create mean probability variable
            img_prob = dst.createVariable("mean_probability_per_img", "f4", ('t'))
            img_prob.units = 'Average prediction confidence per image'
            img_prob[:] = mean_probabilities[:]

            print("Output netCDF file created.")

    # optionally calculate melt pond fraction and store in csv file
    if not args.skip_mpf:
        masks_path = os.path.join(args.predicted_path, "np/", "masks.npy")
        probabilities_path = os.path.join(args.predicted_path, "np/", "probabilities.npy")

        # USED FOR MODEL CONFIDENCE ANALYSIS
        low_conf_threshold = 0.6
        probabilities = np.load(probabilities_path)
    
        pred = probabilities.argmax(axis=1)  # (n_samples, 480, 480)

        mp_probability_per_image = []
        si_probability_per_image = []
        oc_probability_per_image = []
        low_conf_pixels = []

        for i in range(probabilities.shape[0]):
            img_probs = probabilities[i]  # (3, 480, 480)
            img_pred = pred[i]            # (480, 480)

            mp_mask = img_pred == 0
            si_mask = img_pred == 1
            oc_mask = img_pred == 2

            # mean probability of each predicted class
            mp_probability_per_image.append(img_probs[0, mp_mask].mean() if mp_mask.any() else np.nan)
            si_probability_per_image.append(img_probs[1, si_mask].mean() if si_mask.any() else np.nan)
            oc_probability_per_image.append(img_probs[2, oc_mask].mean() if oc_mask.any() else np.nan)

            low_conf_pixels.append((probabilities[i].max(axis=1) < low_conf_threshold).sum())

        mp_probability_per_image = np.array(mp_probability_per_image)
        si_probability_per_image = np.array(si_probability_per_image)
        oc_probability_per_image = np.array(oc_probability_per_image)

        import pdb; pdb.set_trace()
        
        # make sure to not repeat calculations if output nc file was created
        if not args.create_output_nc:
            _, mean_probabilities = filter_and_calculate_mpf(
                masks_path, probabilities_path, 0.88
            )
            fractions = calculate_mpf(masks_path)

        ds = netCDF4.Dataset(args.data)
        lats = ds.variables["lat"]
        lons = ds.variables["lon"]

        headers = ["flight_date", "lat", "lon", "mpf", "ocf", "sif", "mp_probability_per_image", "si_probability_per_image", "oc_probability_per_image", "no_low_confidence_pixels", "mean_probability_per_img"]

        mpf_dir = os.path.join(output_path, "output.csv")

        if os.path.isdir(mpf_dir):
            import shutil
            # If a directory exists with the same name, remove it
            shutil.rmtree(mpf_dir)

        with open(
            mpf_dir, "w", newline=""
        ) as f:
            writer = csv.writer(f)

            # headers in the first row
            if f.tell() == 0:
                writer.writerow(headers)

            for i, (lat, lon) in enumerate(zip(lats, lons)):
                mpf = fractions[i][0]
                ocf = fractions[i][1]
                sif = fractions[i][2]
                mp_prob = mp_probability_per_image[i]
                si_prob = si_probability_per_image[i]
                oc_prob = oc_probability_per_image[i]
                low_conf_pixels_im = (probabilities[i].max(axis=0) < low_conf_threshold).sum()
                writer.writerow([formatted_date, lat, lon, mpf, ocf, sif, mp_prob, si_prob, oc_prob, low_conf_pixels_im, mean_probabilities[i]])

    print("Process ended.")

if __name__ == "__main__":
    main()
