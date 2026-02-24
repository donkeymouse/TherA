<div align="center">
<h1>TherA: Thermal-Aware Visual-Language Prompting for<br>Controllable RGB-to-Thermal Infrared Translation</h1>

[**Dong-Guw Lee**](https://scholar.google.com/citations?user=u6VDnlgAAAAJ&hl=ko)<sup>1*</sup>&emsp;&emsp;[**Tai Hyoung Rhee**](https://scholar.google.com/citations?user=PF8EfdYAAAAJ&hl=en&oi=ao)<sup>1*</sup>&emsp;&emsp;[**Hyunsoo Jang**](https://rpm.snu.ac.kr)<sup>1</sup><br>[**Young-Sik Shin**](https://scholar.google.com/citations?user=gGfBRawAAAAJ&hl=en&oi=ao)<sup>2</sup>&emsp;&emsp;[**Ukcheol Shin**](https://scholar.google.com/citations?user=ZvxI80EAAAAJ&hl=ko&oi=ao)<sup>3</sup>&emsp;&emsp;[**Ayoung Kim**](https://ayoungk.github.io/)<sup>1&dagger;</sup>

<br>
<sup>1 </sup>Seoul National University&emsp;&emsp;&emsp;<sup>2 </sup>Kyungpook National University&emsp;&emsp;&emsp;<sup>3 </sup>KENTECH <br>
<sup>*</sup> Equal Contribution
<br><br>

**CVPR 2026**

<a href='https://williamrheeth.github.io/mirage_web/'><img src='https://img.shields.io/badge/Project_Page-TherA-blue' alt='Project Page'></a>
<a href="[https://rpm.snu.ac.kr](https://arxiv.org/abs/2602.19430)"><img src='https://img.shields.io/badge/arXiv-TherA-red' alt='ArXiv Link'></a>
</div>



## News
- ⚡(2026-02-24): TherA & R2T2 Dataset repo opening
---

![mirage_fig1](assets/method.png)


🔥 **TherA**: Thermal-Aware Controllable RGB→TIR Translation
- 🧠 Thermal-Aware VLM Conditioning
- 🎯 Dual-Level Controllability
- 🌡 Physically Plausible Synthesis
- 🏆 SoTA Performance & Strong Zero-Shot Generalization

📦 **R2T2**: <ins>100k+ Aligned</ins> RGB–TIR–Text Dataset
- 🖼 112,970 Aligned Triplets: RGB image + TIR image + Canonical thermal schema
- 🏙 Scene Diversity: Driving, CCTV, Aerial, Ego-view
- 🌗 Temporal Diversity: Day/Night, Diurnal transitions
- 🌦 Environmental Diversity: Weather, season, illumination variability
- 🧱 Material & Object-Level Annotation with Structured Canonicalization
- 📚 Compiled from 9 aligned RGB-TIR datasets with additional pseudo-aligned pairs

---

## TherA Overview

### Dataset Overview Table

<details>
  <summary>Click to Expand</summary>

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border-bottom: 2px solid black;">Category</th>
    <th style="border-bottom: 2px solid black;">Dataset</th>
    <th style="border-bottom: 2px solid black;">Train</th>
    <th style="border-bottom: 2px solid black;">Test</th>
    <th style="border-bottom: 2px solid black;">Extra</th>
    <th style="border-bottom: 2px solid black;">Scene</th>
    <th style="border-bottom: 2px solid black;">Weather</th>
    <th style="border-bottom: 2px solid black;">Season</th>
    <th style="border-bottom: 2px solid black;">Location</th>
    <th style="border-bottom: 2px solid black;">Avg. Resolution</th>
  </tr>

  <tr style="font-size:90%">
    <td rowspan="8" style="border-right: 1px solid black;">Outdoor</td>
    <td>MS²</td>
    <td>93,746</td>
    <td>18,896</td>
    <td>84,358</td>
    <td>Campus, Urban, Residential</td>
    <td>Clear, Cloudy, Rainy</td>
    <td>Summer</td>
    <td>Korea</td>
    <td>544 × 191</td>
  </tr>

  <tr style="font-size:90%">
    <td>STHeReO</td>
    <td>46,437</td>
    <td>9,745</td>
    <td></td>
    <td>Campus, Suburban</td>
    <td>Clear</td>
    <td>Summer</td>
    <td>Korea</td>
    <td>601 × 245</td>
  </tr>

  <tr style="font-size:90%">
    <td>ViViD</td>
    <td>35,796</td>
    <td>14,597</td>
    <td></td>
    <td>Campus</td>
    <td>Clear, Cloudy</td>
    <td>Spring</td>
    <td>Korea</td>
    <td>629 × 497</td>
  </tr>

  <tr style="font-size:90%">
    <td>NSAVP</td>
    <td>65,333</td>
    <td>78,823</td>
    <td></td>
    <td>Urban, Suburban</td>
    <td>Clear, Cloudy</td>
    <td>Summer</td>
    <td>Korea</td>
    <td>640 × 512</td>
  </tr>

  <tr style="font-size:90%">
    <td>CAMEL (Outdoor)</td>
    <td>8,581</td>
    <td>4,482</td>
    <td></td>
    <td>Campus, Road, Urban</td>
    <td>Clear, Cloudy, Snow</td>
    <td>Spring, Fall, Winter</td>
    <td>USA</td>
    <td>404 × 230</td>
  </tr>

  <tr style="font-size:90%">
    <td>TRI2I</td>
    <td>19,768</td>
    <td>11,913</td>
    <td></td>
    <td>Campus, Road</td>
    <td>Clear, Cloudy</td>
    <td>Spring, Summer</td>
    <td>USA</td>
    <td>229 × 228</td>
  </tr>

  <tr style="font-size:90%">
    <td>METU-VisTIR</td>
    <td>33</td>
    <td>1,052</td>
    <td></td>
    <td>Campus</td>
    <td>Clear</td>
    <td></td>
    <td>Turkey</td>
    <td>632 × 497</td>
  </tr>

  <tr>
    <td><i>MIRAGE Outdoor</i></td>
    <td>269,694</td>
    <td>139,508</td>
    <td>84,358</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr style="font-size:90%">
    <td rowspan="5" style="border-right: 1px solid black;">Indoor</td>
    <td>Trimodal</td>
    <td>4,550</td>
    <td>2,653</td>
    <td></td>
    <td>Room</td>
    <td></td>
    <td></td>
    <td>Austria</td>
    <td>640 × 480</td>
  </tr>

  <tr style="font-size:90%">
    <td>MultiSpectralMotion</td>
    <td>11,575</td>
    <td>5,777</td>
    <td>3,647</td>
    <td>Room</td>
    <td></td>
    <td></td>
    <td>China</td>
    <td>640 × 480</td>
  </tr>

  <tr style="font-size:90%">
    <td>OdomBeyondVision</td>
    <td>20,904</td>
    <td>4,372</td>
    <td></td>
    <td>Room</td>
    <td></td>
    <td></td>
    <td>UK</td>
    <td>328 × 249</td>
  </tr>

  <tr style="font-size:90%">
    <td>CAMEL (Indoor)</td>
    <td>221</td>
    <td>117</td>
    <td></td>
    <td>Hall</td>
    <td></td>
    <td></td>
    <td>USA</td>
    <td>404 × 230</td>
  </tr>

  <tr>
    <td><i>MIRAGE Indoor</i></td>
    <td>37,250</td>
    <td>12,919</td>
    <td>88,005</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Total</td>
    <td><b>MIRAGE</b></td>
    <td>306,944</td>
    <td>152,427</td>
    <td>88,005</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td></td>
    <td><b>MIRAGE Raw</b></td>
    <td>278,341</td>
    <td>130,491</td>
    <td>88,005</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

</table>

- MIRAGE Raw represents the data pairs providing both 8-bit and raw 14-bit TIR
</details>

### Example Visualizations of each Dataset Sequence in MIRAGE (Left: RGB / Right: TIR)

![representatives](assets/MIRAGE_samples.png)


### Comparison with Other Datasets

![comparison](assets/dataset_comparison.png)

MIRAGE not only surpasses existing datasets in terms of scale and diversity, but also provides 8-bit & 14-bit TIR images.

---

## Download

Refer to the link below for dataset download.

[Google Drive](https://rpm.snu.ac.kr/)

---

## Dataset Structure

```
  MIRAGE
  ├── {$DATASET_NAME}
  |   └── {$SEQUENCE_NAME}
  |       ├── RGB
  |       |   ├── 1.jpg
  |       |   └── ...
  |       └── TIR
  |           ├── 1.jpg
  |           └── ...
  ├── ...
  ├── ViVID
  |   ├── img_campus_day1
  |   |   ├── RGB
  |   |   |   ├── 000001.png
  |   |   |   └── ...
  |   |   └── TIR
  |   |       ├── 000001.png
  |   |       └── ...
  |   ├── ...
  ├── ...
  ```

---

## Benchmark Evaluation

### State-of-the-Art Image Translation Model Evaluation

#### RGB-to-TIR ###
<details>
  <summary>Click to Expand</summary>

It's a gif file. Changes every 5 seconds
  
![rgb2tir](https://github.com/user-attachments/assets/c17ff001-8e35-43eb-be29-6d643d114854)

</details>

#### TIR-to-RGB ###

<details>
  <summary>Click to Expand</summary>

It's a gif file. Changes every 5 seconds
  
![tir2rgb](https://github.com/user-attachments/assets/ccfbfe0f-2bf8-4134-8183-b51197c03487)

</details>

### Zero-shot Performance of MIRAGE

<details>
  <summary>Click to Expand</summary>
  
Zero-shot performance of PID trained on MIRAGE in comparison to PID trained with other benchmarks
  
![zero_shot](https://github.com/user-attachments/assets/b99b2a94-0e03-431a-b583-f403c252e92d)

</details>


## TODO List

- [x] Open git repository
- [ ] Upload dataset
- [ ] Upload curation framework
- [ ] add more baselines


---

## Citation
If you found our work useful, please cite
```
@inproceedings{mirage2025,
title={MIRAGE: Large-Scale Aligned RGB-Thermal Infrared Dataset for Scalable Multispectral Translation},
  author={Lee, Dong-Guw and Jang, Hyunsoo and Rhee, TaiHyoung and Shin, UkCheol and Kim, Ayoung},
  booktitle={ArXiv},
  year={2025}
}

}
```
