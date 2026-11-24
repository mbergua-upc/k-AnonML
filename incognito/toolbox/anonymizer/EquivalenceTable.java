package anonymizer;

import java.sql.ResultSet;
import java.sql.SQLException;

import sqlwrapper.QueryResult;
import sqlwrapper.SQLWrapper;
import sqlwrapper.SqLiteSQLWrapper;

/**
 * The class that manages equivalence classes. Typical schema of the table contains: 
 * <ul>
 * 	<li> eid: equivalence identifier
 * 	<li> generalized quasi-identifier attribute values 
 * (vector of intervals, one for each qi-attribute)
 * </ul>
 * 
 */
public class EquivalenceTable {
	/** Quasi-identifier attributes*/
	private QIDAttribute[] qid;
	/** Name of the table*/
	private String tableName;
	/** SQL connection/querying object*/
	private SQLWrapper sqlwrapper;
	
	/**
	 * Class constructor
	 * @param qid Quasi-identifier attribtues
	 * @param tableName Name of the table
	 */
	public EquivalenceTable(QIDAttribute[] qid, String tableName) {
		this.qid = qid;
		this.tableName = tableName;
		sqlwrapper = SqLiteSQLWrapper.getInstance();
		createTable();
	}
	
//	/**
//	 * Class constructor (call only if the table is created before)
//	 * @param name Name of the table to be fetched
//	 * @param copyFrom Equivalence from which the paremeters will be copied
//	 */
//	public EquivalenceTable(String name, EquivalenceTable copyFrom) {
//		this.qid = copyFrom.qid;
//		this.tableName = name;
//		this.sqlwrapper = SqLiteSQLWrapper.getInstance();
//		//no calls to createTable - table already exists
//	}
	
	/**
	 * Creates the table that will store Equivalence objects
	 */
	private void createTable() {
		//first, check if table exists
		String select_SQL = "SELECT NAME FROM SQLITE_MASTER WHERE NAME = '" + tableName + "'";
		QueryResult result = sqlwrapper.executeQuery(select_SQL);
		if(result.hasNext()) { // if the table exists, drop table
			String drop_SQL = "DROP TABLE " + tableName;
			sqlwrapper.execute(drop_SQL);
		}
		
		//tableName(EID BIGINT, ATT_qi1, ..., ATT_qiN, PRIMARY KEY(EID)) 
		String createTable_SQL = "CREATE TABLE " + tableName + 
			" (EID BIGINT PRIMARY KEY, ";
		for(int i = 0; i < qid.length; i++) {
			createTable_SQL += "ATT_" + qid[i].index + " VARCHAR(128), ";
		}
		//trim the last comma
		createTable_SQL = createTable_SQL.substring(0, createTable_SQL.length() - 2); 
		createTable_SQL += ")";
		//execute update
		sqlwrapper.execute(createTable_SQL);
		
		//index on the generalized values, used by getEID (avoids a full scan per tuple)
		String columns = "";
		for(int i = 0; i < qid.length; i++) {
			columns += "ATT_" + qid[i].index + ", ";
		}
		columns = columns.substring(0, columns.length() - 2);
		sqlwrapper.execute("CREATE INDEX IDX_" + tableName + " ON " + tableName + " (" + columns + ")");
	}
	
	/**
	 * Getter for tableName
	 * @return Name of the table
	 */
	public String getName() {
		return tableName;
	}
	
	/**
	 * Get the equivalence ID for the provided set of generalized values
	 * @param genVals String representations of Interval objects (one per QI-attribute)
	 * @return EID of the matching equivalence or -1 if not found
	 */
	public Long getEID(String[] genVals) throws SQLException{
		//validate input
		if(genVals.length != qid.length) {
			return new Long(-1);
		}
		//the selection query, to be issued to the database
		String select_SQL = "SELECT EID FROM " + tableName + " WHERE ";
		for(int i = 0; i < qid.length; i++) {
			select_SQL += "ATT_" + qid[i].index + " ='" + genVals[i] + "' AND ";
		}
		//remove the last AND
		select_SQL = select_SQL.substring(0, select_SQL.length()-4);
		//execute query
		QueryResult result = sqlwrapper.executeQuery(select_SQL);
		if(result.hasNext()) {
			return ((ResultSet) result.next()).getLong(1);
		} else {
			return new Long(-1);
		}
	}
	
	/**
	 * Get the generalization for an equivalence
	 * @param eid Equivalence ID of the equivalence
	 * @return an array of generalized values (i.e., string representation of the intervals)
	 */
	public String[] getGeneralization(double eid) throws SQLException{
		String select_SQL = "SELECT * FROM " + tableName + " WHERE EID = " + eid;		
		//execute query
		QueryResult result = sqlwrapper.executeQuery(select_SQL);
		
		if(result.hasNext()) {
			String[] retVal = new String[qid.length];
			ResultSet rs = (ResultSet) result.next();
			for(int i = 0; i < retVal.length; i++) {
				retVal[i] = rs.getString(i+2); //+1 because rs indices start with 1, +1 to omit the EID column
			}
			return retVal;
		} else {
			return null;
		}
	}
	
	/**
	 * Get the generalization of all the equivalences
	 * @return map from the equivalence ID to the array of generalized values 
	 */
	public java.util.Map<Long, String[]> getAllGeneralizations() throws SQLException{
		java.util.Map<Long, String[]> all = new java.util.HashMap<Long, String[]>();
		QueryResult result = sqlwrapper.executeQuery("SELECT * FROM " + tableName);
		while(result.hasNext()) {
			ResultSet rs = (ResultSet) result.next();
			String[] vals = new String[qid.length];
			for(int i = 0; i < vals.length; i++) {
				vals[i] = rs.getString(i+2);
			}
			all.put(rs.getLong(1), vals);
		}
		return all;
	}
	
	/**
	 * Inserts many new equivalences with a few statements. The caller has to make sure 
	 * that the equivalence IDs are not in the table yet.
	 * @param eids Equivalence IDs of the new equivalences
	 * @param genVals Generalized values of each new equivalence (same order as eids)
	 */
	public void insertEquivalences(java.util.List<Long> eids, java.util.List<String[]> genVals) {
		final int chunk = 200;
		for(int start = 0; start < eids.size(); start += chunk) {
			StringBuilder insert_SQL = new StringBuilder("INSERT INTO " + tableName + " VALUES ");
			int end = Math.min(start + chunk, eids.size());
			for(int j = start; j < end; j++) {
				insert_SQL.append(j > start ? ", (" : "(").append(eids.get(j));
				for(int i = 0; i < genVals.get(j).length; i++) {
					insert_SQL.append(", '").append(genVals.get(j)[i]).append("'");
				}
				insert_SQL.append(")");
			}
			sqlwrapper.execute(insert_SQL.toString());
		}
	}
	
	/**
	 * Overwrites the generalized values of this equivalence with the specified set
	 * of values
	 * @param eid Equivalence id of the equivalence
	 * @param newVals new generalization values
	 */
	public void setGeneralization(double eid, String[] newVals) throws SQLException {
		String update_SQL = "UPDATE " + tableName + " SET ";
		update_SQL += "ATT_" + qid[0].index + " = '" + newVals[0] + "'";
		for(int i = 1; i < qid.length; i++) {
			update_SQL += ", ATT_" + qid[i].index + " = '" + newVals[i] + "'";
		}
		update_SQL += " WHERE EID = " + eid;
		sqlwrapper.execute(update_SQL);
	}
	
	/**
	 * Inserts a nwe tuple
	 * @param vals Values of the tuple, as read from the data source 
	 * (i.e., before any generalization)
	 * @return Equivalence id of the new or existing equivalence
	 * @throws Exception
	 */
	public Long insertTuple(String[] vals) throws Exception {
		//get generalization values
		String[] genVals = new String[qid.length];
		for(int i =0; i < genVals.length; i++) {
			String attVal = vals[qid[i].index];
			if(qid[i].catDomMapping != null) {
				attVal = qid[i].catDomMapping.get(attVal).toString();
			} 
			genVals[i] = new Interval(attVal).toString();
		}
		//check if an equivalence matching genVals already exists
		Long eid = getEID(genVals); 
		if(eid.compareTo(new Long(-1)) == 0) { //if not, insert new equivalence
			eid = insertEquivalence(genVals);
		}
		return eid;
	}
	
	/**
	 * Inserts a new equivalence
	 * @param genVals String representation of generalized values (i.e., generated 
	 * with Interval.toString)
	 * @return Equivalence id of the new or existing equivalence
	 * @throws SQLExpcetion
	 */
	public Long insertEquivalence(String[] genVals) throws SQLException{
		Long eid = new Long(0);
		//get the largest ID on the table
		String select_SQL = "SELECT MAX(EID) FROM " + tableName;
		QueryResult result = sqlwrapper.executeQuery(select_SQL);
		if(result.hasNext()) {
			eid = ((ResultSet) result.next()).getLong(1);
			eid++;
		}
		
		String insert_SQL = "INSERT INTO " + tableName + " VALUES ( " + eid.toString() + ", ";
		for(int i = 0; i < genVals.length; i++) {
			insert_SQL += "'" + genVals[i] + "', ";
		}
		//trim the last comma
		insert_SQL = insert_SQL.substring(0, insert_SQL.length()-2);
		insert_SQL += ")";
		sqlwrapper.execute(insert_SQL);
		return eid;
	}
	
	/**
	 * Deletes the equivalence
	 * @param eid Equivalence ID of the equivalence to be deleted
	 */
	public void deleteEquivalence(Long eid) {
		String delete_SQL = "DELETE FROM " + tableName + " WHERE EID = " + eid;
		sqlwrapper.execute(delete_SQL);
	}
	
	/**
	 * Drops this table from the database
	 */
	public void drop() {
		String drop_SQL = "DROP TABLE " + tableName;
		sqlwrapper.execute(drop_SQL);
	}
}
