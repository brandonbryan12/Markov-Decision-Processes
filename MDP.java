/**
 * Brandon Ferrell
 * Intro to Machine Learning
 * Assignment 3: Markov Decision Processes and Reinforcement Learning
 */

import java.util.*;
import java.io.*;
import java.lang.*;
import java.text.DecimalFormat;
import java.math.RoundingMode;

class MDP {

	// Filename given by user
	String filename;

	// MDP Attributes
	int numStates;
	int numActions;
	Double discountFactor;

	// Map of all of the states
	TreeMap<Integer, State> states = new TreeMap<>();

	public static void main(String[] args) throws Exception {
		MDP mdp = new MDP();

		mdp.cliInput();
		mdp.parseFiles();
		mdp.runIterations(20);
	}

	/**
	* Get filename and other data from user
	*/
	private void cliInput() {
		Scanner in = new Scanner(System.in);

		System.out.print("Enter the number of states in the MDP: ");
		numStates = in.nextInt();

		System.out.print("Enter the number of actions: ");
		numActions = in.nextInt();

		System.out.print("Enter the name of the input file: ");
		filename = in.next();

		System.out.print("Enter the discount factor: ");
		discountFactor = in.nextDouble();
	}

	/**
	* Parse file and store data into necessary data structures
	*/
	private void parseFiles() throws Exception {
		FileInputStream fstream = new FileInputStream(filename);
		BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

		String strLine;
		State s;
		int stateNum;

		// Parse training file
		while((strLine = br.readLine()) != null) {

			// Skip blank lines
			if(strLine.trim().length() == 0) {
				continue;
			}

			strLine = strLine.trim();

			// Store data
			s = new State();

			String[] splitArray = strLine.split("\\s+");
			String[] actionSplitArray;

			stateNum = Integer.parseInt(splitArray[0].substring(1));

			s.num = stateNum;

			s.reward = Integer.parseInt(splitArray[1]);

			String actionsLine = strLine.substring(strLine.indexOf('('));

			ArrayList<String> actionsLines = new ArrayList<>();

			// Break actions into lines
			while(actionsLine.length() > 0) {
				String line = actionsLine.substring(1, actionsLine.indexOf(')'));
				actionsLines.add(line);

				String newLine;

				if(actionsLine.indexOf(')') + 1 < actionsLine.length() - 1) {
					newLine = actionsLine.substring(actionsLine.indexOf(')') + 1);
				}
				else {
					newLine = "";
				}

				actionsLine = newLine.trim();
			}

			// Parse actions
			for(String actionLine : actionsLines) {
				actionSplitArray = actionLine.split("\\s+");

				int actionNum = Integer.parseInt(actionSplitArray[0].substring(1));
				int astateNum = Integer.parseInt(actionSplitArray[1].substring(1));
				double probability = Double.parseDouble(actionSplitArray[2]);

				Action a = s.actions.get(actionNum);

				if(a == null) {
					a = new Action();
				}

				a.num = actionNum;

				a.stateProbabilities.put(astateNum, probability);

				s.actions.put(actionNum, a);

			}

			states.put(stateNum, s);

		}

		br.close();
	}

	/**
	*	Iterate, create the J values for each state along with the optimal policy for n iterations
	*/
	private void runIterations(int iterations) {
		double[][] jMatrix = new double[20][states.size()];

		for(int i = 0; i < iterations; i++) {
			runIteration(i, jMatrix);
			printIterationData(i);

			// Copy new values to matrix
			for(Map.Entry<Integer, State> stateEntry : states.entrySet()) {
				jMatrix[i][stateEntry.getKey() - 1] = stateEntry.getValue().jVal;
			}
		}
	}

	/**
	*	Run the iteration
	*/
	private void runIteration(int i, double[][] jMatrix) {
		for(Map.Entry<Integer, State> stateEntry : states.entrySet()) {
			State state = stateEntry.getValue();

			// Only pick up reward on first iteration
			if(i == 0) {
				state.jVal = state.reward;
			}
			else {
				int bestAction = 0;
				double bestVal = Double.NEGATIVE_INFINITY;

				// Get best weighted sum for the actions
				for(Map.Entry<Integer, Action> actionEntry : state.actions.entrySet()) {
					Action action = actionEntry.getValue();

					double weightedVal = 0.0;

					for(Map.Entry<Integer, Double> probabilityEntry : action.stateProbabilities.entrySet()) {
						double probability = probabilityEntry.getValue();
						weightedVal += (probability * jMatrix[i - 1][probabilityEntry.getKey() - 1]);
					}

					// Update for best sum
					if(weightedVal > bestVal) {
						bestVal = weightedVal;
						bestAction = action.num;
					}
				}

				// set the state's best J value and optimal policy
				state.jVal = state.reward + discountFactor * bestVal;	
				state.bestAction = bestAction;
			}
		}
	}

	/**
	*	Print the optimal policy for each state and their J values
	*/
	private void printIterationData(int i) {
		System.out.println("After iteration " + (i + 1) + ":");

		for(Map.Entry<Integer, State> stateEntry : states.entrySet()) {
			State state = stateEntry.getValue();

			DecimalFormat df = new DecimalFormat("0.0000");
			df.setRoundingMode(RoundingMode.FLOOR);

			System.out.print("(s" + state.num + " a" + state.bestAction + " " + df.format(state.jVal) + ") ");
		}

		System.out.println();
	}

	class State {
		int reward;
		int num;
		double jVal;
		int bestAction;

		HashMap<Integer, Action> actions;

		public State() {
			actions = new HashMap<>();
			jVal = 0;
		}

	}

	class Action {
		int num;

		HashMap<Integer, Double> stateProbabilities;

		public Action() {
			stateProbabilities = new HashMap<>();
		}
	}
}