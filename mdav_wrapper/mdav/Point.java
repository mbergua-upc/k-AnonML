package mdav_wrapper.mdav;

public class Point {
	private double[] numericalValues;
	private String[] qualitativeValues;
	
	
	public Point(double[] numericalValues, String[] qualitativeValues) {
		this.numericalValues = numericalValues;
		this.qualitativeValues = qualitativeValues;
	}
	
	double[] getNumericalValues() {
		return numericalValues;
	}
	
	String[] getQualitativeValues() {
		return qualitativeValues;
	}
}
