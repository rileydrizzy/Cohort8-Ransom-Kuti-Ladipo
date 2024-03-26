# NSL-2-AUDIO

<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/AISaturdaysLagos/Cohort8-Ransome-Kuti-Ladipo">
    <img src=".assets/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">NSL-2-AUDIO</h3>

  <p align="center">
    NSL-2-AUDIO is an open-source Automatic Sign Language Translation system, specifically designed to translate Nigerian Sign Language (NSL) into one of the Low-Resource Languages (LRLs) spoken in Nigeria."
    <br />
    <a href="https://github.com/AISaturdaysLagos/Cohort8-Ransome-Kuti-Ladipo"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/AISaturdaysLagos/Cohort8-Ransome-Kuti-Ladipo">View Demo</a>
    ·
    <a href="https://github.com/AISaturdaysLagos/Cohort8-Ransome-Kuti-Ladipo/issues">Report Bug</a>
    ·
    <a href="https://github.com/AISaturdaysLagos/Cohort8-Ransome-Kuti-Ladipo/issues">Request Feature</a>
  </p>
</div>

> [!IMPORTANT]  
> This project is currently in a very early development/experimental stage. There are a lot of unimplemented/broken features at the moment. Contributions are welcome to help out with the progress!

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [About](#about)
- [Demos](#demos)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Getting Started](#getting-started)
- [Roadmap](#bulb-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

<!-- ABOUT THE PROJECT -->
## About

***Overview:*** \
This project is dedicated to the development of an Automatic Sign Language Translation system, with a specific focus on translating Nigerian Sign Language (NSL) into one of the Low-Resource Languages (LRLs) spoken in Nigeria. The primary objective is to address the communication challenges faced by the NSL community and contribute to inclusivity and employment opportunities in education and the workforce.

***Significance:*** \
Effective communication is a cornerstone of societal cohesion, and this project addresses a critical gap in the integration of the NSL community into the broader society. Sign language, while vital for the hearing-impaired, often faces limitations in bridging the gap with the larger community. This project seeks to overcome these limitations and promote a more inclusive and understanding society.

***Potential Applications:***

- ***Education***:
  - Integration into schools and educational institutions to support hearing-impaired students.
  - Facilitation of communication in educational settings, creating a more inclusive learning environment.
- ***Workforce:***
  - Facilitating communication in the workplace, creating job opportunities for NSL speakers.
  - Promotion of diversity and inclusion in professional environments.
- ***Community Involvement:***
  - Empowering NSL speakers to engage more actively in community activities.
  - Bridging the gap between the NSL community and the broader society.

You can read the project proposal here, [Project Proposal](https://github.com/AISaturdaysLagos/Cohort8-Ransome-Kuti-Ladipo/blob/main/project-proposal.pdf)

### Built With

- [![Python][Python]][Python-url]
- [![Pytorch][Pytorch]][Pytorch-url]
- [![GCP][GCP]][GCP-URL]
- [![MEAT][META]][META-url]
- [![HuggingFace][HuggingFace]][HuggingFace-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

> [!NOTE]
> Currently support only Yoruba

## Demos

> This is a Temporary demo

[![Video Title](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)

## System Architecture

<!--[high_level_system_design](images) -->
### Architecture of Inference Pipeline

![inference_pipeline.gif](.assets/inference_pipeline.gif.gif)

### System Design

![system_design](.assets\system_design)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Quick Start

The easiest way to run the project locally:

<!-- GETTING STARTED -->
## Getting Started

The project is structured into three distinct parts, each housed in separate directories as outlined in the project proposal. The initial phase involves translating sign language into English text, followed by the second phase, which focuses on translating the English text into Yoruba text. The final segment entails taking the translated Yoruba text and converting it into generated Yoruba audio.

The `signa2text` directory is dedicated to the process of translating sign language into English text. Meanwhile, the `linguify_yb` directory serves the purpose of transforming English text into Yoruba text. Finally, the `yb2audio` directory is designated for utilizing the translated audio to generate Yoruba audio.

In `app` directory

To access any of the directories, navigate into the respective directory and adhere to the specified prerequisites in the respective directory README.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

## :bulb: Roadmap

## Contributing

We welcome contributions to enhance NSL-2-AUDIO capabilities and improve its performance. To contribute, please see the `CONTRIBUTING.md` file for steps.

## License

NSL-2-AUDIO is released under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more information.

## Contact

If you have questions or need assistance, feel free to reach out to:

**Name:** **Ipadeola Ezekiel Ladipo**  
**Email:** <ipadeolaoladipo@outlook.com>  
**GitHub:** [@rileydrizzy](https://github.com/rileydrizzy)  
**Linkdeln:** [Ipadeola Ladipo](https://www.linkedin.com/in/ladipo-ipadeola/)

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

I would like to acknowledge the outstanding contributions of :

**Name:** Afonja Tejumade ***(```Mentor```)***  
**Email:** <tejumade.afonja@aisaturdayslagos.com>  
**GitHub:** [@tejuafonja](https://github.com/tejuafonja)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/rileydrizzy/NSL_2_AUDIO.svg?style=for-the-badge
[contributors-url]: https://github.com/rileydrizzy/NSL_2_AUDIO/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/rileydrizzy/NSL_2_AUDIO.svg?style=for-the-badge
[forks-url]: https://github.com/rileydrizzy/NSL_2_AUDIO/network/members
[stars-shield]: https://img.shields.io/github/stars/rileydrizzy/NSL_2_AUDIO.svg?style=for-the-badge
[stars-url]: https://github.com/rileydrizzy/NSL_2_AUDIO/stargazers
[issues-shield]: https://img.shields.io/github/issues/rileydrizzy/NSL_2_AUDIO.svg?style=for-the-badge
[issues-url]: https://github.com/rileydrizzy/NSL_2_AUDIO/issues
[license-shield]: https://img.shields.io/github/license/rileydrizzy/NSL_2_AUDIO.svg?style=for-the-badge
[license-url]: https://github.com/rileydrizzy/NSL_2_AUDIO/blob/master/LICENSE.txt
[Python-url]: <https://www.python.org/>
[Python]: <https://img.shields.io/badge/Python-563D7C?style=for-the-badge&logo=python&logoColor=white>
[Pytorch-url]: <https://pytorch.org/>
[Pytorch]: <https://img.shields.io/badge/PyTorch-0769AD?style=for-the-badge&logo=pytorch&logoColor=white>
[GCP-url]: <https://cloud.google.com/>
[GCP]: <https://img.shields.io/badge/Google-0769AD?style=for-the-badge&logo=googlecloud&logoColor=white>
[HuggingFace-url]: <https://huggingface.co/>
[HuggingFace]: <https://img.shields.io/badge/HuggingFace-DD0031?style=for-the-badge&logo=Huggingface&logoColor=white>
[META-url]: <https://ai.meta.com/>
[META]: <https://img.shields.io/badge/meta-563D7C?style=for-the-badge&logo=meta&logoColor=white>
