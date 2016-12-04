
#include <iostream>
#include "stdlib.h"

#include "GRT/GRT.h"

using namespace GRT;


int main (int argc, const char * argv[])
{
    //Create a new instance of the TimeSeriesClassificationData
    TimeSeriesClassificationData trainingData;
    
    //Set the dimensionality of the data (you need to do this before you can add any samples)
    trainingData.setNumDimensions( 3 );
    
    //You can also give the dataset a name (the name should have no spaces)
    trainingData.setDatasetName("DummyData");
    
    //You can also add some info text about the data
    trainingData.setInfoText("This data contains some dummy timeseries data");
    
    //Here you would record a time series, when you have finished recording the time series then add the training sample to the training data
    UINT gestureLabel = 1;
    MatrixDouble trainingSample;
    
    //For now we will just add 10 x 20 random walk data timeseries
    Random random;
    for(UINT k=0; k<10; k++){//For the number of classes
        gestureLabel = k+1;
        
        //Get the init random walk position for this gesture
        VectorDouble startPos( trainingData.getNumDimensions() );
        for(UINT j=0; j<startPos.size(); j++){
            startPos[j] = random.getRandomNumberUniform(-1.0,1.0);
        }
        
        //Generate the 20 time series
        for(UINT x=0; x<20; x++){
            
            //Clear any previous timeseries
            trainingSample.clear();
            
            //Generate the random walk
            UINT randomWalkLength = random.getRandomNumberInt(90, 110);
            VectorDouble sample = startPos;
            for(UINT i=0; i<randomWalkLength; i++){
                for(UINT j=0; j<startPos.size(); j++){
                    sample[j] += random.getRandomNumberUniform(-0.1,0.1);
                }
                
                //Add the sample to the training sample
                trainingSample.push_back( sample );
            }
            
            //Add the training sample to the dataset
            trainingData.addSample( gestureLabel, trainingSample );
            
        }
    }
    
    //After recording your training data you can then save it to a file
    if( !trainingData.saveDatasetToFile( "TrainingData.txt" ) ){
        cout << "Failed to save dataset to file!\n";
        return EXIT_FAILURE;
    }
    
    //This can then be loaded later
    if( !trainingData.loadDatasetFromFile( "TrainingData.txt" ) ){
        cout << "Failed to load dataset from file!\n";
        return EXIT_FAILURE;
    }
    
    //This is how you can get some stats from the training data
    string datasetName = trainingData.getDatasetName();
    string infoText = trainingData.getInfoText();
    UINT numSamples = trainingData.getNumSamples();
    UINT numDimensions = trainingData.getNumDimensions();
    UINT numClasses = trainingData.getNumClasses();
    
    cout << "Dataset Name: " << datasetName << endl;
    cout << "InfoText: " << infoText << endl;
    cout << "NumberOfSamples: " << numSamples << endl;
    cout << "NumberOfDimensions: " << numDimensions << endl;
    cout << "NumberOfClasses: " << numClasses << endl;
    
    //You can also get the minimum and maximum ranges of the data
    vector< MinMax > ranges = trainingData.getRanges();
    
    cout << "The ranges of the dataset are: \n";
    for(UINT j=0; j<ranges.size(); j++){
        cout << "Dimension: " << j << " Min: " << ranges[j].minValue << " Max: " << ranges[j].maxValue << endl;
    }
    
    //If you want to partition the dataset into a training dataset and a test dataset then you can use the partition function
    //A value of 80 means that 80% of the original data will remain in the training dataset and 20% will be returned as the test dataset
    TimeSeriesClassificationData testData = trainingData.partition( 80 );
    
    //If you have multiple datasets that you want to merge together then use the merge function
    if( !trainingData.merge( testData ) ){
        cout << "Failed to merge datasets!\n";
        return EXIT_FAILURE;
    }
    
    //If you want to run K-Fold cross validation using the dataset then you should first spilt the dataset into K-Folds
    //A value of 10 splits the dataset into 10 folds and the true parameter signals that stratified sampling should be used
    if( !trainingData.spiltDataIntoKFolds( 10, true ) ){
        cout << "Failed to spiltDataIntoKFolds!\n";
        return EXIT_FAILURE;
    }
    
    //After you have called the spilt function you can then get the training and test sets for each fold
    for(UINT foldIndex=0; foldIndex<10; foldIndex++){
        TimeSeriesClassificationData foldTrainingData = trainingData.getTrainingFoldData( foldIndex );
        TimeSeriesClassificationData foldTestingData = trainingData.getTestFoldData( foldIndex );
    }
    
    //If need you can clear any training data that you have recorded
    trainingData.clear();
    
    return EXIT_SUCCESS;
}
