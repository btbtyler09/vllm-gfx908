#include "gfx908_wv_fused_w8_launch.cuh"
template bool hw8_launch_nb<8, 0>(const HW8Args& a, int yt, int un, int lpr, int ks);
template bool hw8_launch_nb<8, 1>(const HW8Args& a, int yt, int un, int lpr, int ks);
template bool hw8_launch_nb<8, 2>(const HW8Args& a, int yt, int un, int lpr, int ks);
