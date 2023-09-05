import numpy as np
import statistics


class K_Nearest_Neighbors:
    def __init__(self, distance_matrix_type) -> None:
        self.distance_matrix_type = distance_matrix_type

    # Calculate the distance between two points

    def get_distanceMatrix(self, trainData_point, x_testData_point):
        # trainData points are features and also target column
        # testData points are features of that point
        dist = 0
        # -1 because we don't want to include target column ✅
        loop_len = len(trainData_point) - 1

        # Euclidean Distance :
        if (self.distance_matrix_type == 'euclidean'):
            for i in range(loop_len):
                dist = dist + \
                    (trainData_point[i] - x_testData_point[i]) ** 2
            euclideanDistance = np.sqrt(dist)
            return euclideanDistance

        # Manhattan Distance :
        if (self.distance_matrix_type == 'manhattan'):
            print(loop_len)
            for i in range(loop_len):
                print(i)

                dist = dist + abs(trainData_point[0] - x_testData_point[0])
            manhattanDistance = dist
            return manhattanDistance

        # find list of Nearest point
    def findNearest_point(self, trainData, testData_Point, k):
        distanceList = []

        # get distance from testPoint to all train Data point
        for trainPoint in trainData:
            dist = self.get_distanceMatrix(trainPoint, testData_Point)
            distanceList.append(trainPoint, dist)  # take both

        # Now sort distance in ascending order
        # their are 2 colum last one distance ,first one point_signature
        distanceList.sort(key=lambda x: x[1])

        # then get only top(smallest distance point) k number of points
        neighborsList = []
        for i in range(k):
            # only class or points are taken
            neighborsList.append(distanceList[i][0])

        return neighborsList

    # Predict the class of new data point

    def predict(self, trainData, X_testData, k):
        # Find nearest points
        nearestPoints = self.findNearest_point(trainData, X_testData, k)

        # get the labels of nearest points
        label = []
        for point in nearestPoints:
            label.append(point[-1])

        # then get mode of points
        predictive_class = statistics.mode(label)
        return predictive_class
