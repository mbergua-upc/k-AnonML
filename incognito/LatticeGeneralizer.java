package incognito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.sql.ResultSet;

import sqlwrapper.QueryResult;
import sqlwrapper.SqLiteSQLWrapper;
import anonymizer.AnonRecordTable;
import anonymizer.Configuration;
import anonymizer.EquivalenceTable;

/**
 * Generalizes the original tables according to a lattice entry. It does what the generalize 
 * methods of Incognito_K, Incognito_L and Incognito_T did, one equivalence at a time with
 * several SQL statements each (a lot of time for the thousands of lattice entries of a real 
 * dataset), but in memory and with a few SQL statements for the whole table:
 * <ul>
 * <li> the generalized values of the equivalences of the original table are read only once,
 * <li> every generalization of a value (a hierarchy lookup) is stored the first time it is computed,
 * <li> the new equivalences are inserted together and the records are copied with a single INSERT,
 * <li> the tables of an entry that has an equivalence smaller than requiredClassSize are not filled.
 * </ul>
 * The tables are the same as before (up to the numbering of the equivalences).
 */
public class LatticeGeneralizer {
	/**
	 * Smallest size that the equivalences of a lattice entry are allowed to have. If an entry 
	 * has a smaller equivalence, its tables are not filled (it can not satisfy a privacy 
	 * definition that includes k-anonymity, as set by IncognitoRunner), which is much faster. 
	 * 0 fills the tables of every entry. 
	 */
	public static int requiredClassSize = 0;

	private final Configuration conf;
	private final EquivalenceTable origET;
	private final AnonRecordTable origAT;

	/** Equivalences of the original table, in the order of their IDs */
	private final Map<Long, String[]> original;

	/** Number of records of every equivalence of the original table */
	private final Map<Long, Integer> originalSizes = new HashMap<Long, Integer>();

	/** For every quasi-identifier, from a value to its successive generalizations (the value itself first) */
	private final List<Map<String, List<String>>> chains = new ArrayList<Map<String, List<String>>>();

	/**
	 * Class constructor
	 * @param conf Configuration instance
	 * @param origET Equivalence table of the original data
	 * @param origAT Anonymized records table of the original data
	 */
	public LatticeGeneralizer(Configuration conf, EquivalenceTable origET, AnonRecordTable origAT) throws Exception {
		this.conf = conf;
		this.origET = origET;
		this.origAT = origAT;
		this.original = new LinkedHashMap<Long, String[]>(origET.getAllGeneralizations());
		QueryResult sizes = SqLiteSQLWrapper.getInstance().executeQuery(
				"SELECT EID, COUNT(*) FROM " + origAT.getName() + " GROUP BY EID");
		while (sizes.hasNext()) {
			ResultSet rs = (ResultSet) sizes.next();
			originalSizes.put(rs.getLong(1), rs.getInt(2));
		}
		for (int i = 0; i < conf.qidAtts.length; i++) {
			chains.add(new HashMap<String, List<String>>());
		}
	}

	private String generalize(int attribute, String value, int height) {
		List<String> chain = chains.get(attribute).get(value);
		if (chain == null) {
			chain = new ArrayList<String>();
			chain.add(value);
			chains.get(attribute).put(value, chain);
		}
		while (chain.size() <= height) {
			chain.add(conf.qidAtts[attribute].generalize(chain.get(chain.size() - 1)));
		}
		return chain.get(height);
	}

	/**
	 * Fills the tables of a lattice entry from the original tables
	 * @param root An entry of the generalization lattice that specifies how many 
	 * times each qi-attribute will be generalized
	 * @param currET New (empty) equivalence table of the entry
	 * @param currAT New (empty) anonymized records table of the entry
	 * @return false (and leaves the tables empty) if some equivalence is smaller than 
	 * requiredClassSize, true otherwise
	 */
	public boolean generalize(LatticeEntry root, EquivalenceTable currET, AnonRecordTable currAT) throws Exception {
		Map<String, Long> newEIDs = new HashMap<String, Long>();
		List<Long> eids = new ArrayList<Long>();
		List<String[]> genVals = new ArrayList<String[]>();
		List<Integer> sizes = new ArrayList<Integer>();
		Map<Long, Long> mapping = new HashMap<Long, Long>();

		for (Map.Entry<Long, String[]> entry : original.entrySet()) {
			String[] vals = new String[conf.qidAtts.length];
			StringBuilder key = new StringBuilder();
			for (int i = 0; i < vals.length; i++) {
				vals[i] = generalize(i, entry.getValue()[i], root.heightAt(i));
				key.append(vals[i]).append('\u0001');
			}
			Long newEID = newEIDs.get(key.toString());
			if (newEID == null) {
				newEID = Long.valueOf(eids.size() + 1);
				newEIDs.put(key.toString(), newEID);
				eids.add(newEID);
				genVals.add(vals);
				sizes.add(0);
			}
			int index = newEID.intValue() - 1;
			sizes.set(index, sizes.get(index) + originalSizes.get(entry.getKey()));
			mapping.put(entry.getKey(), newEID);
		}

		for (int size : sizes) {
			if (size < requiredClassSize) {
				return false;
			}
		}

		currET.insertEquivalences(eids, genVals);
		currAT.copyFromMapped(origAT, mapping);
		return true;
	}
}
