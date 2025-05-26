# Create the datasets directory if it doesn't exist
mkdir -p datasets

dataset=$1

# If a folder with the name of the dataset already exists, exit
if [ -d "datasets/$dataset" ]; then
    echo "Dataset already downloaded."
    exit 1
fi

case $dataset in
    cosmology)
        curl -o datasets/cosmology.zip https://zenodo.org/api/records/11479419/files-archive
        ;;
    polymer)
        curl -o datasets/polymer.zip https://zenodo.org/api/records/6764836/files-archive
        ;;
    # eagle)
    #     curl -o datasets/eagle.zip https://zenodo.org/api/records/11479419/files-archive
    #     ;;
    # shapenet)
    #     curl -o datasets/shapenet.zip https://zenodo.org/api/records/11479419/files-archive
    #     ;;
    *)
        echo "Invalid dataset. Please choose from: cosmology, polymer."
        # echo "Invalid dataset. Please choose from: cosmology, polymer, eagle, shapenet."
        exit 1
esac

for file in datasets/*.zip; do
    # Unzip the file
    unzip -o $file -d "${file%.zip}"
    # Remove the zip file
    rm $file
done
