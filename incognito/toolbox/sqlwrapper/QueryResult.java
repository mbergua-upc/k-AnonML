package sqlwrapper;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Implementation of the query result used by the toolbox. It is a modified version of the
 * original, which read the result of the query page by page and left the statement open until
 * the last record was read: with the current SQLite driver an open statement locks its tables,
 * and the callers that read only the first record (e.g., to get a maximum or a count) made
 * the following DROP TABLE fail. Here the whole result is read into memory and the statement
 * is closed in the constructor.
 * <p>
 * next() returns an object that implements ResultSet and is positioned on the next record;
 * only the getters used by the toolbox (getLong, getInt, getDouble, getString and getObject, by
 * column index starting with 1 or by column name) are supported.
 */
public class QueryResult implements Iterator<Object> {

	private final List<Object[]> rows = new ArrayList<Object[]>();
	private final Map<String, Integer> columns = new HashMap<String, Integer>();
	private int currentRecordInd = 0;

	/**
	 * Class constructor
	 * @param stat Statement object for executing sql query (closed by the constructor)
	 * @param sql SQL query
	 */
	public QueryResult(Statement stat, String sql) throws SQLException {
		try {
			ResultSet rs = stat.executeQuery(sql);
			ResultSetMetaData metaData = rs.getMetaData();
			int numCols = metaData.getColumnCount();
			for (int i = 1; i <= numCols; i++) {
				columns.put(metaData.getColumnLabel(i).toUpperCase(Locale.ROOT), i - 1);
			}
			while (rs.next()) {
				Object[] row = new Object[numCols];
				for (int i = 0; i < numCols; i++) {
					row[i] = rs.getObject(i + 1);
				}
				rows.add(row);
			}
			rs.close();
		} finally {
			stat.close();
		}
	}

	/**
	 * Returns false if there are no more records, true otherwise.
	 * @return success
	 */
	public boolean hasNext() {
		return currentRecordInd < rows.size();
	}

	/**
	 * Extract the next record from the result.
	 * @return the next record, as a ResultSet
	 */
	public Object next() {
		final Object[] row = rows.get(currentRecordInd++);
		InvocationHandler handler = new InvocationHandler() {
			public Object invoke(Object proxy, java.lang.reflect.Method method, Object[] args) {
				String name = method.getName();
				if (name.equals("close")) {
					return null;
				}
				if (args == null || args.length != 1 || !name.startsWith("get")) {
					throw new UnsupportedOperationException(name);
				}
				int index = (args[0] instanceof Integer)
						? ((Integer) args[0]) - 1
						: columns.get(((String) args[0]).toUpperCase(Locale.ROOT));
				Object value = row[index];
				if (name.equals("getObject")) {
					return value;
				} else if (name.equals("getString")) {
					return value == null ? null : value.toString();
				} else if (name.equals("getLong")) {
					return value == null ? 0L : ((Number) value).longValue();
				} else if (name.equals("getInt")) {
					return value == null ? 0 : ((Number) value).intValue();
				} else if (name.equals("getDouble")) {
					return value == null ? 0.0 : ((Number) value).doubleValue();
				}
				throw new UnsupportedOperationException(name);
			}
		};
		return Proxy.newProxyInstance(ResultSet.class.getClassLoader(), new Class<?>[] { ResultSet.class }, handler);
	}

	public void remove() {
	}
}
