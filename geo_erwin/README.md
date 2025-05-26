<div align="center">
<h3>Erwin goes Geometric: Building Erwin to be Equivariant</h4>
<table>
    <tr>
        <td style="width: 100px"><b>Key systems:</b><br><br></td>
        <td>
            <a href="https://github.com/maxxxzdn/erwin">Erwin: A Tree-based Hierarchical Transformer</a>,<br>
            <a href="https://github.com/Qualcomm-AI-research/geometric-algebra-transformer">Geometric Algebra Transformer</a> (GATr)
        </td>
    </tr>
    <tr>
        <td style="width: 100px"><b>Authors:</b></td>
        <td>
            Gerben Koopman, Emo Maat, Tijs Wiegman, Önder Akaçık
        </td>
    </tr>
</table>
</div>

## Introduction
In this work, we build Erwin to be equivariant to geometric transformations.

### Erwin
Erwin is a tree-based hierarchical transformer that uses a ball tree to compute attention. The ball tree is built recursively, and the attention is computed within the balls of the tree. This allows for efficient computation of attention, as the balls can be processed in parallel. The model is designed to be computationally efficient and can be used for various tasks, including image classification and natural language processing.

### Geometric Algebra Transformer (GATr)
The Geometric Algebra Transformer (GATr) is a transformer designed to be equivariant to geometric transformations, such as rotations and translations. It uses geometric algebra to represent the input data and compute attention. This allows the model to be more robust to changes in the input data, as it can learn to recognize patterns regardless of their orientation or position.

## Contribution
The main drawback of the Erwin model is that it is not equivariant to geometric transformations. In this work, we extend the Erwin model to be equivariant to geometric transformations by using a geometric algebra transformer (GATr). The GATr is a transformer designed to be equivariant to geometric transformations, such as rotations and translations.

<h4>By combining the Erwin model with GATr, we create a model that is equivariant to geometric transformations.</h4>

### Running the code
In order to run the code, you will need conda installed. This environment required to run the project is defined in the `environment.yml` file. This uses cuda 12.0 (`cu120`), and this might need changing depending on your driver version.

Just like with the original Erwin code, running the project is done by first downloading the
```bash
# Download the data using wget
mkdir datasets && cd datasets
wget https://zenodo.org/api/records/11479419/files-archive -O zenodo_data.zip
unzip zenodo_data.zip -d cosmolody/
cd ..

# Run the code on the cosmology dataset
cd experiments
python train_cosmology.py --data-path ../datasets/cosmology
```

## Structure of project
To understand the code, it is beneficial to understand the original Erwin code, as the code resembles the original Erwin code closely. This project adheres to the source material, with minimal changes to make the model geometric using the GATr library. Consequently, the main changes occur in `models/erwin.py` and `models/mpnn.py`.