# Yoruba  to Audio-Loom (***yb2audio***)

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11.2-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)](https://pytorch.org/)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-description">Project description</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Project description

***Overview:*** \
**Project Description: Generative AI for Communication**

In our ongoing pursuit of advancing communication accessibility for Nigerian Sign Language (NSL) community, the project now enters a critical phase dedicated to the development of a generative model. This specialized model is tasked with seamlessly transforming translated yoruba text into expressive spoken words, bridging the gap between written communication and the auditory landscape. This ambitious endeavor aims to create a more immersive and inclusive user experience by bringing the native linguistic nuances to life through spoken expressions.

**Key Components:**

1. **Generative Model Development:**
   - Rigorous training of a specialized model for eloquent articulation of translated yoruba text.

2. **Utilizing Massively Multilingual Speech (MMS) Model:**
   - Leveraging the pre-existing Massively Multilingual Speech model by [Meta](https://huggingface.co/facebook/mms-tts-yor).

3. **Fine-Tuning with IròyìnSpeech Dataset:**
   - Drawing upon the [IròyìnSpeech dataset](https://arxiv.org/abs/2307.16071) for fine-tuning the MMS model. Leading to a meticulous adaptation to the intricacies of the task, ensuring contextually rich and articulate spoken output.

4. ***Efficient Training and Inferecne***
   - Leveraging a several  of techniques to optimize the efficiency of training, inference, and the comprehensive utilization of computational resources.

**Future Implications:**
The successful implementation of this generative AI approach holds promising implications for various applications not only to our main project, including but not limited to improved accessibility in education, enhanced communication tools for diverse linguistic communities, and a more inclusive digital landscape.

## Getting Started

### Prerequisites

```bash
# Clone this repository
$ git clone 

# Install dependencies
$ make setup

# Go into the repository
$ cd yb2audio
```

### Data

```bash
# Tun to download the dataset
$ python src/dataset/download_dataset.py

```

### TR

```bash
python src/
```

## Contact

**Name:** **Ipadeola Ezekiel Ladipo**  
**Email:** <ipadeolaoladipo@outlook.com>  
**GitHub:** [@rileydrizzy](https://github.com/rileydrizzy)  
**Linkdeln:** [Ipadeola Ladipo](https://www.linkedin.com/in/ladipo-ipadeola/)

## Acknowledgments

I would like to acknowledge the outstanding contributions of :

**Name:** Afonja Tejumade ***(```Mentor```)***  
**Email:** <tejumade.afonja@aisaturdayslagos.com>  
**GitHub:** [@tejuafonja](https://github.com/tejuafonja)
