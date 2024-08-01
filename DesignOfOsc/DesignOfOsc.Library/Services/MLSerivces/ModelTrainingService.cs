using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Microsoft.ML.AutoML.AutoMLExperiment;
using static Microsoft.ML.DataOperationsCatalog;

namespace DesignOfOsc.Library.Services.MLSerivces
{
    public class ModelTrainingService : BaseProxy<ModelTrainingService>
    {
        public async Task ModelInit(string path)
        {
            // Initialize MLContext
            MLContext ctx = new MLContext();

            // Define data path
            var dataPath = Path.GetFullPath(path);

            // Infer column information
            ColumnInferenceResults columnInference =
                ctx.Auto().InferColumns(dataPath, labelColumnName: "fare_amount", groupColumns: false);

            // Create text loader
            TextLoader loader = ctx.Data.CreateTextLoader(columnInference.TextLoaderOptions);

            // Load data into IDataView
            IDataView data = loader.Load(dataPath);

            TrainTestData trainValidationData = ctx.Data.TrainTestSplit(data, testFraction: 0.2);

            SweepablePipeline pipeline = ctx.Auto()
                .Featurizer(data, columnInformation: columnInference.ColumnInformation)
                .Append(ctx.Auto().Regression(labelColumnName: columnInference.ColumnInformation.LabelColumnName));

            // Create AutoML experiment
            var experimentSettings = new MulticlassExperimentSettings()
            {
                MaxExperimentTimeInSeconds = 3600, // 设置最大实验时间  
                //MaxIterationCount = 10, // 设置最大迭代次数  
                //TrainerKinds = new TrainerKinds() // 自定义包含的训练器  
                //{
                //    LightGbm = true,
                //    SdcaLogisticRegression = true,
                //    // 你可以添加或移除你想要的模型  
                //},
                 Trainers 
                ,OptimizingMetric = MulticlassClassificationMetric.TopKAccuracy // 优化指标  
            };

            AutoMLExperimentSettings settings = new AutoMLExperimentSettings()
            {

            };
            
            //AutoMLExperiment experiment = ctx.Auto().CreateMulticlassClassificationExperiment(experimentSettings);
            AutoMLExperiment experiment = ctx.Auto().CreateExperiment();

            // Configure experiment
            experiment
                .SetPipeline(pipeline)
                .SetRegressionMetric(RegressionMetric.RSquared, labelColumn: columnInference.ColumnInformation.LabelColumnName)
                .SetTrainingTimeInSeconds(60)
                .SetDataset(trainValidationData);

            // Log experiment trials
            ctx.Log += (_, e) => {
                if (e.Source.Equals("AutoMLExperiment"))
                {
                    Console.WriteLine(e.RawMessage);
                }
            };

            // Run experiment
            TrialResult experimentResults = await experiment.RunAsync();

            // Get best model
            var model = experimentResults.Model;
        }
        
        public void ModelOldInit()
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
