package mdav_wrapper.mdav;


public class MdavClusteringAlgo {

	public void computeClusters(MdavData dataSet, int k) {
		
		int clustersDone = 0;
		
		long remainingRows = dataSet.rowCount();
		
		double lastCalc = System.currentTimeMillis();
		while ( remainingRows >= 3 * k) {
			
			// Step1: average record 
			Point avgPoint = dataSet.computeAverages();
			
			// find most distant record xr to the average (using euclidian distance)
			Point mostDistantFromAvg = dataSet.findMostDistantFrom(avgPoint);
			
			// step 2: most distant record xs from xr
			Point mostDistandFromRow = dataSet.findMostDistantFrom(mostDistantFromAvg);
			
			// step 3: form two groups around xr and xs:
			// - One group contains xr + (k-1) closest to xr => size k
			// - One group contains xs + (k-1) closest to xs => size k
			dataSet.computeClusterNearPoint(mostDistantFromAvg, k);
			dataSet.computeClusterNearPoint(mostDistandFromRow, k);
			
			clustersDone += 2;
			
			// step 4: workingDataset - (groupNearXr + groupNearXs)
			remainingRows -= 2 * k;
			
			if (clustersDone%100 == 0 ) {
				long newCalc = System.currentTimeMillis();
				System.out.println("Cluster done :" + clustersDone +  " in " + (newCalc - lastCalc) + " ms remaining " + remainingRows);
				lastCalc = newCalc;
			}
			
		}
		
		// step 5 
		if ( remainingRows > 2 * k ) {
			//dataSet.computeAverages();
			//int xdist = dataSet.findMostDistantRecordFromAvg();
			Point avgEnd = dataSet.computeAverages();
			dataSet.computeClusterNearPoint(avgEnd, k);
			
		}
		
		// remaing points are in last group ...
		dataSet.computeLastGroup();

	}
	
}
