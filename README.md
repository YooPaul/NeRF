# Simple Implementation of NeRF (Work In Progress...)
[![Open NeRF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YooPaul/NeRF/blob/master/NeRF.ipynb)<br>

### TODOs

* Conversion to NDC space
* Optimize pipeline, make better use of batch computation

### Dataset

I used images of the [South Building](https://colmap.github.io/datasets.html) and their respective estimated camera poses from SfM [2, 3].   

## References

[1] Mildenhall, Ben, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. "Nerf: Representing scenes as neural radiance fields for view synthesis." In European conference on computer vision, pp. 405-421. Springer, Cham, 2020. 

[2] Schönberger, Johannes Lutz, and Jan-Michael, Frahm. "Structure-from-Motion Revisited." . In Conference on Computer Vision and Pattern Recognition (CVPR).2016.

[3] Schönberger, Johannes Lutz, Enliang, Zheng, Marc, Pollefeys, and Jan-Michael, Frahm. "Pixelwise View Selection for Unstructured Multi-View Stereo." . In European Conference on Computer Vision (ECCV).2016.


