package mdav_wrapper.mdav;

public class IndexValue implements Comparable<IndexValue> {
	private int index;
	private double value;
	
	public IndexValue(int index, double value ) {
		this.index = index;
		this.value = value;
	}
	
	public int getIndex() {
		return this.index;
	}
	
	public double getValue() {
		return this.value;
	}


	@Override
	public int compareTo(IndexValue other) {
		return Double.compare(this.value, other.value);
	}
	
	public String toString() {
		return this.index + ":" + this.value;
	}
}
