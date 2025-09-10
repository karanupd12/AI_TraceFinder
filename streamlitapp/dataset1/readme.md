# Multi-Scale Tampering Maps Dataset

Contact: **Paweł Korus**, Shenzhen University and AGH University of Science and Technology
E-mail: pkorus [at] agh [dot] edu [dot] pl
Web: http://kt.agh.edu.pl/~korus
Github: https://github.com/pkorus/multiscale-prnu

## Description

This dataset contains multi-scale tampering probability maps obtained by running a standard sliding-window PRNU detector (with central-pixel attribution) on a set of 136 tampered images. The images represent realistic forgeries, created by hand in modern photo-editing software (GIMP and Affinity Photo). The images were captured by four different cameras: Sony alpha57 (own dataset), Canon 60D (courtesy of dr Bin Li), Nikon D7000, Nikon D90 (RAISE dataset http://mmlab.disi.unitn.it/RAISE/).

The dataset is intended for testing multi-scale fusion algorithms. Each forgery case has the following data:
- seven 240 x 135 px multi-scale tampering probability maps (stored in `.mat` files)
- 480 x 270 px ground truth tampering map (`_mask.png` files)
- 480 x 270 px RGB thumbnail of the tampered image (`.tif` files)

A preview of an example forgery is shown below.

![Example forgery](./preview.png)

**Notes:**

- In order to minimize the size of the dataset, the tampering probability maps are stored with `uint16` precision. Use Matlab's function `im2double` to obtain floating-point numbers in range [0,1].

- The thumbnails can be used to guide the localization / fusion schemes. Full scale images (1920 x 1080 px) are available as a separate dataset (see http://kt.agh.edu.pl/~korus/downloads/dataset-realistic-tampering/).

## Using the Dataset

The dataset comes from our paper *"Multi-scale Analysis Strategies in PRNU-based Tampering Localization"* and can be used only for educational and research purposes. If you use it in your research, please cite the following paper:

- P. Korus & J. Huang, Multi-scale Analysis Strategies in PRNU-based Tampering Localization, IEEE Trans. Information Forensics & Security, 2017

For Bibtex users:

```
@article{Korus2016TIFS,
  Author = {P. Korus and J. Huang},
  Journal = {IEEE Trans. on Information Forensics \& Security},
  Title = {Multi-scale Analysis Strategies in PRNU-based Tampering Localization},
  Year = {2017}
}
```

The dataset can be easily used in conjunction with our multi-scale analysis toolbox. You can obtain a copy as follows:

```
# git clone https://github.com/pkorus/multiscale-prnu
```

Once you have cloned the repository, you can easily download this dataset using:

```
# ./configure.py data:maps
```

Example use of these maps can be found in the `demo_fusion` script. A simple benchmark is also available (`demo_benchmark` script). For more information, please refer to the documentation of the toolbox and the above-mentioned paper.
