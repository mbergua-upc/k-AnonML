package mdav_wrapper.mdav;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Command line front end for MDAV, called from mdav_wrapper/anonymizer.py.
 *
 * It reads a numeric matrix (one record per line, comma separated, no header),
 * partitions the records into clusters of at least k records with MDAV and writes
 * the cluster id of each record (same order as the input) to the output file, one
 * id per line. All columns of the matrix are used to compute distances.
 *
 * Usage: java -cp build mdav.MdavRunner --input matrix.csv --output clusters.txt --k 5
 */
public class MdavRunner {

	public static void main(String[] args) throws IOException {
		Map<String, String> opt = new HashMap<>();
		for (int i = 0; i + 1 < args.length; i += 2) {
			opt.put(args[i], args[i + 1]);
		}
		for (String required : new String[] { "--input", "--output", "--k" }) {
			if (!opt.containsKey(required)) {
				System.err.println("Missing argument " + required);
				System.exit(2);
			}
		}

		int k = Integer.parseInt(opt.get("--k"));
		double[][] values = readMatrix(opt.get("--input"));
		if (k < 1 || values.length < k) {
			System.err.println("Invalid k=" + k + " for " + values.length + " records");
			System.exit(2);
		}

		int[] allColumns = new int[values[0].length];
		for (int i = 0; i < allColumns.length; i++) {
			allColumns[i] = i;
		}

		MdavNumericData data = new MdavNumericData(values, allColumns, 1);
		new MdavClusteringAlgo().computeClusters(data, k);

		try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(opt.get("--output")), StandardCharsets.UTF_8)) {
			for (int cluster : data.getClusters()) {
				writer.write(Integer.toString(cluster));
				writer.newLine();
			}
		}
	}

	private static double[][] readMatrix(String path) throws IOException {
		List<double[]> rows = new ArrayList<>();
		try (BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
			String line;
			while ((line = reader.readLine()) != null) {
				if (line.isEmpty()) {
					continue;
				}
				String[] parts = line.split(",");
				double[] row = new double[parts.length];
				for (int i = 0; i < parts.length; i++) {
					row[i] = Double.parseDouble(parts[i]);
				}
				rows.add(row);
			}
		}
		if (rows.isEmpty()) {
			System.err.println("Empty input matrix");
			System.exit(2);
		}
		return rows.toArray(new double[0][]);
	}
}
