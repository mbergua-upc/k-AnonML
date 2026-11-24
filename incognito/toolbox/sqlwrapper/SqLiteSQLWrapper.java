package sqlwrapper;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * Implementation of the SQLWrapper interface for SQLite embedded database. 
 * 
 */

public class SqLiteSQLWrapper implements SQLWrapper {

    // The original driver (SQLite.JDBCDriver, needs the native library sqlite_jni) was replaced by
    // org.xerial:sqlite-jdbc, and the database is kept in memory instead of ~/.Toolbox/toolbox.dat,
    // so that several processes can run at the same time and nothing is left on disk.
    private final String driver = "org.sqlite.JDBC";
    private final String url = "jdbc:sqlite::memory:";
	
	private Connection conn = null;
    
	/**
	 * Single instance created upon class loading.
	 */
	 
	private static final SqLiteSQLWrapper sqLiteInstance = new SqLiteSQLWrapper();

    private SqLiteSQLWrapper() {	    	
    }
    
	/**
	 * Returns the singleton sqLite instance
	 * @return sqLiteInstance
	 */
    public static SqLiteSQLWrapper getInstance() {
    	
    	try{
    		if(sqLiteInstance.conn == null || sqLiteInstance.conn.isClosed())
    			sqLiteInstance.conn = sqLiteInstance.getConnection();
    		
    	}catch(Exception ex){
    		ex.printStackTrace();
    		return null;
    	}
        
        return sqLiteInstance;
     }

	/**
	 * Execute SQL statement for data definition and manipulation
	 * @param sql Sql operation
	 * @return success of executed operation
	 */
    
	public boolean execute(String sql) {
		Statement st = null;
		try {
			st =sqLiteInstance.conn.createStatement();
			boolean successful = st.execute(sql);
			st.close();
			return successful;
		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		} finally {
			try {
				st.close();
			} catch(Exception e) {}
		}
	}

	/**
	 * Execute SQL statement for data query
	 * @param sql Sql operation
	 * @return queryResult
	 */
	
	public QueryResult executeQuery(String sql) {

		QueryResult result = null;
		
		try {
			Statement st =sqLiteInstance.conn.createStatement();
			result = new QueryResult(st, sql);			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return result;
	}

	
	/**
	 * Commit transaction
	 */
	
	public void commit() {
		try {
			conn.commit();
		} catch(SQLException e) {
			e.printStackTrace();
		}
	}

	
	/**
	 * Commit transaction, delete temporary data
	 * @return success of the operation 
	 */
	
	public boolean flush() {
		try {
			conn.commit();
			conn.close();
		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}		
		return true;
	}
	
	private Connection getConnection() throws SQLException {
		 
	    loadDriver();
	    
	    conn = DriverManager.getConnection(sqLiteInstance.url);
	    conn.setAutoCommit(false);
	
		return conn;
	}

    private void loadDriver() throws SQLException {
        try {
            Class.forName(sqLiteInstance.driver);
         
        } catch (ClassNotFoundException cnfe) {
            System.err.println("\nUnable to load the JDBC driver ");
            System.err.println("Please check your CLASSPATH.");
            cnfe.printStackTrace(System.err);
        } 
    }
}
