import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXConfiguration.AnonymizationAlgorithm;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.criteria.DistinctLDiversity;
import org.deidentifier.arx.criteria.EntropyLDiversity;
import org.deidentifier.arx.criteria.EqualDistanceTCloseness;
import org.deidentifier.arx.criteria.KAnonymity;

/**
 * Command line front end for the ARX library, called from arx_wrapper/anonymizer.py.
 *
 * It reads a CSV dataset, anonymizes it with ARX and writes the result to another
 * CSV file:
 *   1. Every column is set as insensitive, then the quasi identifiers (QI) get
 *      their generalization hierarchy loaded from CSV files, and the sensitive
 *      attribute is declared when l-diversity or t-closeness is requested.
 *   2. The privacy models are added: k-anonymity always, plus distinct or entropy
 *      l-diversity and equal-distance t-closeness if requested.
 *   3. The suppression limit and the search algorithm are configured and ARX is
 *      run. If no solution satisfies the models, it exits with code 3.
 *   4. The anonymized data is saved to the output file.
 *
 * Usage: java -cp arx.jar ArxRunner.java --input in.csv --output out.csv --k 5 --algorithm OPTIMAL
 *            --hierarchy-prefix path/dataset --qi name1,name2 [options]
 *
 * Required:
 *   --input, --output   CSV files with ';' as delimiter and a header row
 *   --k                 k-anonymity
 *   --algorithm         name of ARXConfiguration.AnonymizationAlgorithm
 *   --hierarchy-prefix  the hierarchy of a QI is read from prefix + "_hierarchy_" + name + ".csv"
 *   --qi                comma separated names of the quasi identifiers
 * Optional:
 *   --suppression       maximum fraction of suppressed records (0.0 - 1.0), default 0
 *   --sensitive         name of the sensitive attribute (required by l-diversity and t-closeness)
 *   --l                 l-diversity parameter
 *   --l-variant         "distinct" (default) or "entropy"
 *   --t                 t-closeness parameter (equal distance, 0.0 - 1.0)
 */
public class ArxRunner {

    private static final char DELIMITER = ';';

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = new HashMap<>();
        for (int i = 0; i + 1 < args.length; i += 2) {
            opt.put(args[i], args[i + 1]);
        }
        for (String required : new String[] { "--input", "--output", "--k", "--algorithm", "--hierarchy-prefix", "--qi" }) {
            
            if (!opt.containsKey(required)) {
                System.err.println("Missing argument " + required);
                System.exit(2);
            }
        }

        int k = Integer.parseInt(opt.get("--k"));
        double suppression = Double.parseDouble(opt.getOrDefault("--suppression", "0"));
        AnonymizationAlgorithm algorithm = AnonymizationAlgorithm.valueOf(opt.get("--algorithm"));
        String sensitive = opt.get("--sensitive");
        if ((opt.containsKey("--l") || opt.containsKey("--t")) && sensitive == null) {
            System.err.println("--sensitive is required for l-diversity and t-closeness");
            System.exit(2);
        }

        Data data = Data.create(new File(opt.get("--input")), StandardCharsets.UTF_8, DELIMITER);
        for (int i = 0; i < data.getHandle().getNumColumns(); i++) {
            data.getDefinition().setAttributeType(data.getHandle().getAttributeName(i), AttributeType.INSENSITIVE_ATTRIBUTE);
        }
        for (String qi : opt.get("--qi").split(",")) {
            File hierarchyFile = new File(opt.get("--hierarchy-prefix") + "_hierarchy_" + qi + ".csv");
            Hierarchy hierarchy = Hierarchy.create(hierarchyFile, StandardCharsets.UTF_8, DELIMITER);
            data.getDefinition().setAttributeType(qi, hierarchy);
        }
        if (opt.containsKey("--l") || opt.containsKey("--t")) {
            data.getDefinition().setAttributeType(sensitive, AttributeType.SENSITIVE_ATTRIBUTE);
        }

        ARXConfiguration config = ARXConfiguration.create();
        config.addPrivacyModel(new KAnonymity(k));
        if (opt.containsKey("--l")) {
            String variant = opt.getOrDefault("--l-variant", "distinct");
            if (variant.equals("distinct")) {
                config.addPrivacyModel(new DistinctLDiversity(sensitive, (int) Double.parseDouble(opt.get("--l"))));
            } else if (variant.equals("entropy")) {
                config.addPrivacyModel(new EntropyLDiversity(sensitive, Double.parseDouble(opt.get("--l"))));
            } else {
                System.err.println("Unknown l-diversity variant " + variant);
                System.exit(2);
            }
        }
        if (opt.containsKey("--t")) {
            config.addPrivacyModel(new EqualDistanceTCloseness(sensitive, Double.parseDouble(opt.get("--t"))));
        }
        config.setSuppressionLimit(suppression);
        config.setAlgorithm(algorithm);

        ARXResult result = new ARXAnonymizer().anonymize(data, config);
        if (!result.isResultAvailable()) {
            System.err.println("ARX found no solution for the given privacy models");
            System.exit(3);
        }
        result.getOutput().save(new File(opt.get("--output")), DELIMITER);
    }
}
