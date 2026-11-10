import java.io.File;
import java.nio.charset.StandardCharsets;

import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXConfiguration.AnonymizationAlgorithm;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.criteria.KAnonymity;

/**
 * Minimal command line front end for the ARX library.
 *
 * Usage: java -cp arx.jar ArxRunner.java input.csv output.csv k suppression algorithm hierarchyPrefix qi1,qi2,...
 *
 * - input.csv / output.csv use ';' as delimiter and contain a header row.
 * - suppression is the maximum fraction of suppressed records (0.0 - 1.0).
 * - algorithm is a name of ARXConfiguration.AnonymizationAlgorithm.
 * - the hierarchy of a quasi identifier is read from hierarchyPrefix + "_hierarchy_" + name + ".csv".
 */
public class ArxRunner {

    private static final char DELIMITER = ';';

    public static void main(String[] args) throws Exception {
        if (args.length != 7) {
            System.err.println("Usage: ArxRunner input output k suppression algorithm hierarchyPrefix qiNames");
            System.exit(2);
        }
        File input = new File(args[0]);
        File output = new File(args[1]);
        int k = Integer.parseInt(args[2]);
        double suppression = Double.parseDouble(args[3]);
        AnonymizationAlgorithm algorithm = AnonymizationAlgorithm.valueOf(args[4]);
        String hierarchyPrefix = args[5];
        String[] qiNames = args[6].split(",");

        Data data = Data.create(input, StandardCharsets.UTF_8, DELIMITER);
        for (int i = 0; i < data.getHandle().getNumColumns(); i++) {
            data.getDefinition().setAttributeType(data.getHandle().getAttributeName(i), AttributeType.INSENSITIVE_ATTRIBUTE);
        }
        for (String qi : qiNames) {
            File hierarchyFile = new File(hierarchyPrefix + "_hierarchy_" + qi + ".csv");
            Hierarchy hierarchy = Hierarchy.create(hierarchyFile, StandardCharsets.UTF_8, DELIMITER);
            data.getDefinition().setAttributeType(qi, hierarchy);
        }

        ARXConfiguration config = ARXConfiguration.create();
        config.addPrivacyModel(new KAnonymity(k));
        config.setSuppressionLimit(suppression);
        config.setAlgorithm(algorithm);

        ARXResult result = new ARXAnonymizer().anonymize(data, config);
        if (!result.isResultAvailable()) {
            System.err.println("ARX found no solution for k=" + k);
            System.exit(3);
        }
        DataHandle handle = result.getOutput();
        handle.save(output, DELIMITER);
    }
}
