package mdav_wrapper.mdav;


/**
 * Define interface  
 */
public interface MdavData {

	long rowCount();

	Point computeAverages();

	String displayClusters();

	void displayClustersByLine();
	
	Point findMostDistantFrom(Point data);

	void computeClusterNearPoint(Point point, int k);
	
	int[] getClusters();
	
	int getCurrentClusterId();

	void computeLastGroup();

}