import streamlit as st
import numpy as np
import pickle
import math
from kaggle_environments import make
from testEnv import play
from MCTSNode import MCTSNode, validActions

# --- CONFIGURATION & MCTS UTILS ---
c = 1.41  # Exploration constant for UCB1

def move(nodde):
    maxAction = []
    for i in nodde.children:
        if i == False:
            maxAct = -1 * np.inf
        elif i == True:
            maxAct = 0
        elif i.N == 0:
            maxAct = np.inf
        else:
            maxAct = i.Q + c * math.sqrt(math.log(nodde.N) / i.N)
        maxAction.append(maxAct)
    return int(np.argmax(maxAction)), maxAction

@st.cache_resource
def load_mcts_tree(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
# --- SESSION STATE INITIALIZATION ---
if 'initialized' not in st.session_state:
    # 1. Load the Tree
    st.session_state.root_node = load_mcts_tree("mcts_tree.pkl")

    # 2. Setup Environment
    st.session_state.connect = make("connectx", debug=True)
    st.session_state.currNode = st.session_state.root_node
    st.session_state.agentPlayer = None
    st.session_state.turn = 0  # 0 is Red, 1 is Yellow
    st.session_state.game_over = False
    st.session_state.winner_text = ""
    st.session_state.initialized = True
    st.session_state.turn_text = ""


# --- HELPER FUNCTIONS ---
def save_tree(): #CORRECT CODE
    global plays
    temp_node = st.session_state.currNode
    while temp_node.parent is not None:
        temp_node = temp_node.parent
    with open("mcts_tree.pkl", "wb") as f:
        pickle.dump(temp_node, f)
    play()


def check_game_end(): #CORRECT CODE
    env = st.session_state.connect
    if env.done:
        st.session_state.game_over = True
        reward = env.state[st.session_state.agentPlayer]["reward"]
        if reward == 1:
            st.session_state.winner_text = "You Lost!"
        elif reward == -1:
            st.session_state.winner_text = "You Win!"
        else:
            st.session_state.winner_text = "Draw"
        save_tree()
        return True
    return False


def execute_agent_turn(): #CORRECT CODE
    currNode = st.session_state.currNode
    turn = st.session_state.turn

    # Update tree if children aren't evaluated
    if [i for i in currNode.children if i == True]:
        st.write("Agent is learning before making a move. Could take around a minute...")
        for _ in range(7):
            currNode.treeTraverse()


    agentMove, agentActionList = move(currNode)
    agentAction = [None, None]
    agentAction[turn] = int(agentMove)

    st.session_state.connect.step(agentAction)
    st.session_state.currNode = currNode.children[agentMove]
    st.session_state.turn = abs(turn - 1)

    check_game_end()


# --- UI LAYOUT ---
titleCol, buttonCol = st.columns([0.9,0.1])
with titleCol:
    st.title("Connect 4: MCTS Intelligence")
with buttonCol:
    with st.popover(label="", icon=":material/info:"):
        st.markdown("### Rules")
        st.markdown("***Taking Turns:*** Players take turns dropping one of their discs into any of the seven columns.")
        st.markdown("***Gravity Rules:*** Because the board is vertical, your disc will always fall to the lowest available space within that column. You cannot float a piece in the middle of the board.")
        st.markdown("***Blocking:*** On your turn, you must decide whether to advance your own line of four or drop a disc to block your opponent from completing their line.")
        st.markdown("### Ending the Game")
        st.markdown("***Winning:*** The game ends immediately when a player connects four discs.")
        st.markdown("***A Draw (Stalemate):*** If all 42 spaces on the board are filled and no one has a line of four, the game is a draw. This is sometimes called a Full Board")
        st.markdown("### Controls")
        st.markdown("Click on one of the buttons to drop a piece into the corresponding column")

# 1. Initial Selection
if st.session_state.agentPlayer is None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔴 Choose Red (You go first)"):
            st.session_state.turn_text = "Your turn"
            st.session_state.agentPlayer = 1  # Agent is Yellow
            st.rerun()
    with col2:
        if st.button("🟡 Choose Yellow (Agent goes first)"):
            st.session_state.turn_text = "Your turn"
            st.session_state.agentPlayer = 0  # Agent is Red
            execute_agent_turn()  # Kick off the first move
            st.rerun()
    st.write(f"**{st.session_state.turn_text}**")


# 3. Game Interaction
else:
    # Display Winner Message
    if st.session_state.game_over:
        st.header(st.session_state.winner_text)
        with open("plays.txt","r") as f:
            plays = f.read()
        st.markdown(f"## Model has now been trained over {plays} games.")
        if st.button("Restart Game"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Board Display
    board = np.array(st.session_state.connect.state[0]['observation']['board']).reshape(6, 7)


    def get_symbol(val):
        if val == 1: return "🔴"
        if val == 2: return "🟡"
        return "⚪"


    display_grid = [[get_symbol(cell) for cell in row] for row in board]
    st.table(display_grid)

    # Move Buttons (Player's Turn)
    if not st.session_state.game_over and st.session_state.turn != st.session_state.agentPlayer:
        st.write(f"**{st.session_state.turn_text}**")
        st.session_state.turn_text = "Agent's Turn"
        cols = st.columns(7)
        for i in range(7):
            if cols[i].button(f"{i}", key=f"drop_{i}"):
                # Player Move
                playerAction = [None, None]
                playerAction[st.session_state.turn] = i
                st.session_state.connect.step(playerAction)

                # Update Tree based on player move

                if st.session_state.currNode.children[i] == True:
                    st.write("Agent is learning based on player's move. Could take around a minute...")
                    for _ in range(7):
                        st.session_state.currNode.treeTraverse()

                st.session_state.currNode = st.session_state.currNode.children[i]
                st.session_state.turn = abs(st.session_state.turn - 1)

                # Immediately check if player won
                check_game_end()
                st.rerun()

    # Automated transition if it's the Agent's turn (Edge case handling)
    if not st.session_state.game_over and st.session_state.turn == st.session_state.agentPlayer:
        st.write(f"**{st.session_state.turn_text}**")
        st.session_state.turn_text = "Your turn"
        execute_agent_turn()
        st.rerun()