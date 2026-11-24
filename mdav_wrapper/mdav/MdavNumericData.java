package mdav_wrapper.mdav;

import java.util.Arrays;
import java.util.Collections;
import java.util.PriorityQueue;

/**
 * Implementation with all variables numerical (double) 
 *
 */
public class MdavNumericData implements MdavData {
	
	private double[][] values;
	
	private int[] clusters;
	private int rowCount;
	private int colCount;
	
	// numerical cols
	private int[] numericalColIndexes;
	
	// cluster id for next group
	private int currentClusterId;

	
	public MdavNumericData( double[][] values, int[] numericalColIndexes, int startingClusterId) {
		this.values = values;
		
		this.rowCount = values.length;
		this.colCount = values[0].length;
		this.clusters = new int[rowCount];
		this.numericalColIndexes = numericalColIndexes;
		this.currentClusterId = startingClusterId;
	}
	
	/* (non-Javadoc)
	 * @see mdav.MdavClusturing#rowCount()
	 */
	@Override
	public long rowCount() {
		return rowCount;
	}

	public int[] getClusters() {
		return clusters;
	}
	
	/* (non-Javadoc)
	 * @see mdav.MdavClusturing#computeAverages()
	 */
	@Override
	public Point computeAverages() {
		double[] averages = new double[colCount];
		int rowsUsed = 0;
		for ( int rowIdx = 0; rowIdx < rowCount; rowIdx ++ ) {
			// Only data not clustered
			if ( clusters[rowIdx] == 0 ) { 
				for (int colIdx : numericalColIndexes ) {
					averages[colIdx] += values[rowIdx][colIdx]; 
				}
				rowsUsed++;
			}
		}
		
		for (int colIdx : numericalColIndexes ) {
			averages[colIdx] /= rowsUsed;
		}
		
		return new Point(averages, null);
	}

	
	public String toString() {
		return Arrays.toString(values);
	}
	
	/* (non-Javadoc)
	 * @see mdav.MdavClusturing#displayClusters()
	 */
	@Override
	public String displayClusters() {
		return Arrays.toString(clusters);
	}
	
	/* (non-Javadoc)
	 * @see mdav.MdavClusturing#displayClustersByLine()
	 */
	@Override
	public void displayClustersByLine() {
		for ( int cluster : clusters ) {
			System.out.println(cluster);
		}
	}
	
	@Override
	public Point findMostDistantFrom(Point data) {
		
		int mostDistantRow = -1; 
		double maxDistance = 0.;
		
		for ( int rowIdx = 0; rowIdx < rowCount; rowIdx ++ ) {
			// Only data not clustered
			if ( clusters[rowIdx] == 0 ) {
				//countNotclusters++;
				double distance = calcDistance(rowIdx, data);
				if ( distance >= maxDistance ) {
					mostDistantRow = rowIdx;
					maxDistance = distance;
				}
			}
		
		}
		
		return new Point( values[mostDistantRow], null);
	}


	private double calcDistance(int rowIdx, Point to) {
		double distance = 0.;
		double[] toVals = to.getNumericalValues();
		for (int colIdx : numericalColIndexes ) {
			distance += Math.pow(toVals[colIdx] - values[rowIdx][colIdx], 2.);
		}
		
		return distance;
	}

	/* (non-Javadoc)
	 * @see mdav.MdavClusturing#computeClusterNearPoint(int, int)
	 */
	public void computeClusterNearPointOld(Point point, int k) {
		
		// max-heap bounded to k elements: keeps the k nearest points
		PriorityQueue<IndexValue> actualNearestPoint = new PriorityQueue<>(k + 1, Collections.reverseOrder());
		
		for (int rowIdx = 0; rowIdx < rowCount; rowIdx++ ) {
			 if (clusters[rowIdx] == 0 ) {
				 // compute distance
				 actualNearestPoint.add(new IndexValue(rowIdx, calcDistance(rowIdx, point)));
				 if ( actualNearestPoint.size() > k ) {
					 actualNearestPoint.poll();
				 }
			 }
		}
		
		// Set cluster
		for (IndexValue val : actualNearestPoint) {
			clusters[val.getIndex()] = currentClusterId; 
		}
		
		currentClusterId++;
		
	}
	
	public void computeClusterNearPoint(Point point, int k) {
		
		// max-heap bounded to k elements: the head is the farthest of the k nearest points
		PriorityQueue<IndexValue> actualNearestPoint = new PriorityQueue<>(k + 1, Collections.reverseOrder());
		
		for (int rowIdx = 0; rowIdx < rowCount; rowIdx++ ) {
			 if (clusters[rowIdx] == 0 ) {
				 
				 double distance = calcDistance(rowIdx, point);
				 
				 if ( actualNearestPoint.size() < k ) {
					 actualNearestPoint.add(new IndexValue(rowIdx, distance));
				 }
				 else if ( distance < actualNearestPoint.peek().getValue() ) {
					 actualNearestPoint.poll();
					 actualNearestPoint.add(new IndexValue(rowIdx, distance));
				 }
			 }
		}
		
		// Set cluster
		for (IndexValue val : actualNearestPoint) {
			clusters[val.getIndex()] = currentClusterId; 
		}
		
		currentClusterId++;
		
	}


	@Override
	public int getCurrentClusterId() {
		return this.currentClusterId;
	}

	@Override
	public void computeLastGroup() {
		// replace 0 by new group
		for (int rowIdx = 0; rowIdx < rowCount; rowIdx++ ) {
			 if (clusters[rowIdx] == 0 ) {
				 clusters[rowIdx] = currentClusterId; 
			 }
		}
		
		currentClusterId++;
	}



}
