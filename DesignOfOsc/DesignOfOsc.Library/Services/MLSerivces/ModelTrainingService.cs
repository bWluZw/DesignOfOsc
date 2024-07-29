using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;

namespace DesignOfOsc.Library.Services.MLSerivces
{
    internal class ModelTrainingService : BaseProxy<ModelTrainingService>
    {
        public void ModelInit()
        {
            ////Step 1. Create an ML Context
            //var ctx = new MLContext();

            ////Step 2. Read in the input data from a text file for model training
            //IDataView trainingData = ctx.Data
            //    .LoadFromTextFile<ModelInput>(dataPath, hasHeader: true);

            ////Step 3. Build your data processing and training pipeline
            //var pipeline = ctx.Transforms.Text
            //    .FeaturizeText("Features", nameof(SentimentIssue.Text))
            //    .Append(ctx.BinaryClassification.Trainers
            //        .LbfgsLogisticRegression("Label", "Features"));

            ////Step 4. Train your model
            //ITransformer trainedModel = pipeline.Fit(trainingData);

            ////Step 5. Make predictions using your trained model
            //var predictionEngine = ctx.Model
            //    .CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            //var sampleStatement = new ModelInput() { Text = "This is a horrible movie" };

            //var prediction = predictionEngine.Predict(sampleStatement);
        }
    }
}
