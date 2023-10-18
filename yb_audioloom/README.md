# Yoruba Audio-Loom (***yb_audioloom***)

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0-orange)](https://pytorch.org/)

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

## Configuration and Usage

```bash
# Clone this repository
$ git clone

# Go into the repository
$ cd 

# Install dependencies
$ make setup

# activate virtual enviroment
$ source $(poetry env info --path)/bin/activate
```

## Acknowledgments

I would like to acknowledge the outstanding contributions of :

**Name:** Afonja Tejumade ***(```Mentor```)***  
**Email:** <tejumade.afonja@aisaturdayslagos.com>  
**GitHub:** [Afonja Tejumade](https://github.com/tejuafonja)

**Name:** Fola Animashaun ***(```Mentor```)***  
**Email:** <afolabianimashaun@yahoo.co.uk>  
**GitHub:** [Fola Animashaun](https://github.com/Modinat-A)

## Contact

**Name:** Ipadeola Ladipo  
**Email:** <ipadeolaoladipo@outlook.com>  
**GitHub:** [Ipadeoa Ladipo](https://github.com/rileydrizzy)  
**Linkdeln:** [Ipadeoa Ladipo](https://www.linkedin.com/in/ladipo-ipadeola/)
