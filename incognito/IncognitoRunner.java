package incognito;

import anonymizer.AnonRecordTable;
import anonymizer.Anonymizer;
import anonymizer.Configuration;

/**
 * Command line front end for the Incognito implementations of the UTD Anonymization Toolbox
 * (Incognito_K, Incognito_L and Incognito_T), called from incognito/anonymizer.py.
 *
 * It reads the XML configuration file of the toolbox (method incognito_k, incognito_l or
 * incognito_t, parameters k, l, c and t, input and output files, quasi-identifiers with their
 * generalization hierarchies and the sensitive attribute), anonymizes the input and writes
 * the result to the output file of the configuration.
 *
 * Incognito_L and Incognito_T only check l-diversity and t-closeness. As in the rest of the
 * experiments, k-anonymity is always required, so for those methods a table must satisfy
 * k-anonymity and the l-diversity (or t-closeness) requirement.
 *
 * Usage: java -cp build:lib/sqlite-jdbc.jar incognito.IncognitoRunner config.xml
 * Exit codes: 2 invalid arguments or configuration, 3 no generalization satisfies the privacy models.
 */
public class IncognitoRunner {

	/** l-diversity on top of k-anonymity */
	private static class KAnonymousLDiversity extends Incognito_L {
		KAnonymousLDiversity(Configuration conf) throws Exception {
			super(conf);
		}

		@Override
		public boolean satisfiesPrivacyDef(AnonRecordTable table) throws Exception {
			return table.checkKAnonymityRequirement(conf.k) && super.satisfiesPrivacyDef(table);
		}
	}

	/** t-closeness on top of k-anonymity */
	private static class KAnonymousTCloseness extends Incognito_T {
		KAnonymousTCloseness(Configuration conf) throws Exception {
			super(conf);
		}

		@Override
		public boolean satisfiesPrivacyDef(AnonRecordTable table) throws Exception {
			return table.checkKAnonymityRequirement(conf.k) && super.satisfiesPrivacyDef(table);
		}
	}

	public static void main(String[] args) {
		if (args.length != 1) {
			System.err.println("Usage: IncognitoRunner config.xml");
			System.exit(2);
		}

		Anonymizer anon = null;
		try {
			Configuration conf = new Configuration(args[0]);
			// all the methods require k-anonymity: entries with smaller equivalences are discarded without filling their tables
			LatticeGeneralizer.requiredClassSize = conf.k;
			switch (conf.anonMethod) {
			case Configuration.METHOD_INCOGNITO_K:
				anon = new Incognito_K(conf);
				break;
			case Configuration.METHOD_INCOGNITO_L:
				anon = new KAnonymousLDiversity(conf);
				break;
			case Configuration.METHOD_INCOGNITO_T:
				anon = new KAnonymousTCloseness(conf);
				break;
			default:
				throw new Exception("Method not available, use incognito_k, incognito_l or incognito_t");
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(2);
		}

		try {
			anon.readData();
			anon.anonymize();
			anon.outputResults();
		} catch (Exception e) {
			System.err.println("Incognito failed: " + e.getMessage());
			System.exit(3);
		}
	}
}
