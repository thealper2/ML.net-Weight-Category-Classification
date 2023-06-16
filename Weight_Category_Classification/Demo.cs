using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Weight_Category_Classification
{
    internal class Demo
    {
        public static void Execute()
        {
            // Create MLContext
            MLContext context = new MLContext();

            // Data Path
            var path = "C:\\Users\\akrc2\\Downloads\\500_Person_Gender_Height_Weight_Index.csv";

            // Load Data
            var data = context.Data.LoadFromTextFile<InputModel>(path: path, hasHeader: true, separatorChar: ',');

            // Prepare data & create pipeline
            var pipeline = context.Transforms.SelectColumns(
                nameof(InputModel.Gender), nameof(InputModel.Weight), nameof(InputModel.Height), nameof(InputModel.Index))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_Gender", nameof(InputModel.Gender)))
                .Append(context.Transforms.DropColumns(nameof(InputModel.Gender)))
                .Append(context.Transforms.Concatenate("Features", "Encoded_Gender", nameof(InputModel.Height), nameof(InputModel.Weight)))
                .Append(context.Transforms.Conversion.MapValueToKey("Label", nameof(InputModel.Index)))
                .Append(context.MulticlassClassification.Trainers.SdcaNonCalibrated())
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
                
            // Split the data into training and testing sets
            var dataSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // Train the model
            var model = pipeline.Fit(dataSplit.TrainSet);

            // Evaluate the model
            var predictions = model.Transform(dataSplit.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Micro Accuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss}");

            // Make a prediction
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);
            var testData = new InputModel { Gender = "Male", Height = 174, Weight = 96 };
            var prediction = predictionEngine.Predict(testData);

            Console.WriteLine($"Predicted Index: {prediction.Prediction}");

            var save_path = "C:\\Users\\akrc2\\OneDrive\\Masaüstü\\ML.net - Weight Category Classification\\WeighCategory.zip";

            using (var fileStream = new FileStream(save_path, FileMode.Create))
            {
                context.Model.Save(model, data.Schema, fileStream);
            }
        }
    }
}
