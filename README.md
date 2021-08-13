# Malaria Detection  

This project is a proof of concept, where we use a 3D printed [Openflexure microscope](https://openflexure.org/) to take pictures and then detect malaria. We use [NIH images](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets) to train our model. 

The app also functions without an openflexure microscope just by uploading images.

## Demo

![Malaria app Demo](malatec_mvp2.gif)  

## Start-up instructions
Run the docker file from a laptop/ computer while the Openflexure microscope is up and running. First build the dockerfile:

```bash
docker build -t malatec_app .
```
Now you can run the docker file, which starts up the streamlit app:

```bash
docker run -it --rm --name malatec_app -p 8501:8501 -v /Users/fight/Documents/Malaria/malatec_app malatec_app:latest
```
Follow the link provided in the console and you should be able to see the streamlit app.

**Pipeline**    

- Cell segmentation from microscope images using [Article](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1) | [GitHub](https://github.com/MouseLand/cellpose)  
- Classification: SqueezeNet + VGG19 [Article](https://peerj.com/articles/6977.pdf) | [GitHub](https://github.com/sivaramakrishnan-rajaraman/Deep-Neural-Ensembles-toward-Malaria-Parasite-Detection-in-Thin-Blood-Smear-Images)
- Powered by PyTorch, Tensorflow and [Streamlit](https://docs.streamlit.io/en/stable/api.html)  

**[Publicly Available Malria Datasets](https://github.com/danielbarco/malaria_datasets)**

Made with ❤️ in Switzerland ⛰️
  
