using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using static ImageClassificationPractice.Helpers;

namespace ImageClassificationPractice
{
    // Class to represent image input data.
    public class ImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    // Extended data class that holds the loaded image bytes.
    public class ImageDataWithBytes : ImageData
    {
        public byte[] Image { get; set; }
    }

    // Prediction class.
    public class ImagePrediction : ImageData
    {
        public string PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }

    class Program
    {
        // Use the application's base directory combined with the "data" folder.
        static readonly string DataDirectory = Path.Combine(AppContext.BaseDirectory, "data");

        // Where to save the trained model.
        static readonly string ModelPath = Path.Combine(AppContext.BaseDirectory, "model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 1);

            // Load image file paths and labels (only from subfolders)
            IEnumerable<ImageData> images = LoadImagesFromDirectory(DataDirectory, useFolderNameAsLabel: true);
            IDataView data = mlContext.Data.LoadFromEnumerable(images);

            // Split data into training and test sets.
            DataOperationsCatalog.TrainTestData trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Custom mapping: load image bytes from file paths.
            Action<ImageData, ImageDataWithBytes> loadImageBytes = (input, output) =>
            {
                output.ImagePath = input.ImagePath;
                output.Label = input.Label;
                output.Image = File.ReadAllBytes(input.ImagePath);
            };

            // Provide a non-empty contract name.
            EstimatorChain<CustomMappingTransformer<ImageData, ImageDataWithBytes>>? customMapping = mlContext.Transforms.CustomMapping<ImageData, ImageDataWithBytes>(
                loadImageBytes, contractName: "LoadImageBytesMapping")
                // Append a cache checkpoint to break the chain for saving.
                .AppendCacheCheckpoint(mlContext);

            // Build the training pipeline.
            EstimatorChain<KeyToValueMappingTransformer>? pipeline = customMapping
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: nameof(ImageData.Label),
                    outputColumnName: "LabelKey"))
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(
                    featureColumnName: "Image",
                    labelColumnName: "LabelKey"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    inputColumnName: "PredictedLabel",
                    outputColumnName: "PredictedLabel"));

            Console.WriteLine("Training the model...");
            TransformerChain<KeyToValueMappingTransformer>? model = pipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("Evaluating the model...");
            IDataView predictions = model.Transform(trainTestData.TestSet);
            MulticlassClassificationMetrics? metrics = mlContext.MulticlassClassification.Evaluate(
                predictions,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");

            // Save the model.
            mlContext.Model.Save(model, data.Schema, ModelPath);
            Console.WriteLine($"Model saved to {ModelPath}");

            // Test prediction on a single image (sample.png in the root data folder).
            //TODO: Change this to a folder full of samples and predict their labels. 
            PredictionEngine<ImageData, ImagePrediction>? predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            ImageData sampleImage = new ImageData { ImagePath = Path.Combine(DataDirectory, "sample.png"), Label = "" };
            ImagePrediction? prediction = predictor.Predict(sampleImage);
            Console.WriteLine($"Predicted label for sample image: {prediction.PredictedLabel}");
        }
    }
}
