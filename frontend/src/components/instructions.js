import React, { useState } from "react";
import useCollapse from 'react-collapsed'
import styles from "./Instructions.module.css"

function Instructions() {
  // The component that shows the instructions to the user.
  // the instructions are collapsed by default, and can be expanded by clicking the button.

  const [isExpanded, setExpanded] = useState(false)
  const { getCollapseProps, getToggleProps } = useCollapse({ isExpanded })

  return (
    <div className={styles.instructions}>
      <button className={styles.instructionsButton }
        {...getToggleProps({
          onClick: () => setExpanded((prevExpanded) => !prevExpanded),
        })}
      >
        {isExpanded ? 'Instructions ▼' : 'Instructions ►'}
      </button>
      <section {...getCollapseProps()}>
      <h2>Instructions</h2>
      <div className={styles.instructionsBody}>
      <b>Interactive Evolutionary Gallery</b> is a visual art project inspired
      by <a target="_blank" rel="noreferrer" href="https://en.wikipedia.org/wiki/Evolutionary_algorithm">Evolutionary Algorithms</a> and <a target="_blank" rel="noreferrer" href="http://eplex.cs.ucf.edu/papers/secretan_ecj11.pdf">Picbreeder</a>.
      <br /><br />
      The images above each represent an <i>individual</i> in a <i>population</i> of <i>genomes</i>. The population <i>evolves</i> over <i>generations</i> as some individuals survive to reproduce and others do not.
      <br /><br />
      Genomes are <i>Compositional Pattern Producing Networks (<a target="_blank" rel="noreferrer" href="https://en.wikipedia.org/wiki/Compositional_pattern-producing_network">CPPNs</a>)</i>.
      A CPPN contains <i>nodes</i> and <i>connections</i> and is evaluated similar to an artificial neural network to produce an image called the <i>phenotype</i>. Nodes contain activation functions, which determine how that node adds to the phenotype.
      Connections connect nodes and have weights, which determine how much the node at the beginning of the connection contributes to the image. The CPPN is like the DNA of an individual.
      <br/><br/>
      Your job is to decide which individuals survive to reproduce. Click to select your favorite images in the population and then click <b>Next generation</b> to see the results. The individuals that you selected
      will survive unchanged in the next generation. The rest of the population is replaced with the offspring of the survivors.
      <br/><br/>
      Some offspring are produced via <i>asexual reproduction</i>. The remainder are produced via <i>crossover</i>.
      Asexual reproduction produces offspring through <i>mutation</i> of an individual's genome. Crossover combines the genomes of two parents to create a new genome.
      Choose the ratio of offspring created by crossover vs. asexual reproduction in the settings.
      <br/><br/>
      Mutations are small changes to a genome during reproduction. In the case of a CPPN, we can add connections or nodes to the network or remove them from the network.
      Adding nodes/connections increases the complexity of the phenotype image, while removing them reduces complexity. Mutations can also change a node's activation function or the weight of a connection. You can change the rate of the various mutations in the settings.

      <br/><br/>
      If you advance a generation and decide you'd rather go back to try again, press the <b>Previous generation</b> button. Images can be saved to your computer at a higher resolution by selecting the individuals and pressing <b>Save images</b>.
      Press <b>Reset</b> to start over with a new random population.
      </div>
      </section>
    </div>
    )
  }
export default Instructions;
