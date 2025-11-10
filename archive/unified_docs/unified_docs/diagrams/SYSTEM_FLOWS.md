# 📊 System Flow Diagrams

## 🎯 Complete System Flow Overview

### **Unified Trading Engine - High Level Architecture**

```mermaid
graph TB
    subgraph "USER INTERFACE"
        CLI[Command Line Interface]
        CONFIG[unified_config.yaml]
    end

    subgraph "UNIFIED TRADING ENGINE"
        subgraph "Core Engine"
            UTE[Unified Trading Engine]
            CM[Configuration Manager]
            SR[Strategy Registry]
        end

        subgraph "Universal Components"
            US[Universal Strategy Interface]
            MC[Market Context]
            SS[Strategy State]
            PM[Portfolio Manager]
            RM[Risk Manager]
        end

        subgraph "Execution Adapters"
            BA[Backtester Adapter]
            LA[Live Trading Adapter]
        end
    end

    subgraph "EXISTING SYSTEMS"
        subgraph "Backtester System"
            direction TB
            BT_DATA[(CSV Data Files)]
            BT_ENGINE[Backtesting Engine]
            BT_EXEC[Trade Executor]
            BT_STATS[Statistics Calculator]
        end

        subgraph "Live Trading System"
            direction TB
            LT_STREAM[Real-time Data Stream]
            LT_SYSTEM[Trading System]
            LT_ORDERS[Order Executor]
            LT_POS[Position Manager]
            LT_BROKER[Broker API]
        end
    end

    CLI --> CONFIG
    CONFIG --> CM
    CM --> UTE
    UTE --> SR
    SR --> US
    UTE --> BA
    UTE --> LA

    US --> MC
    US --> SS
    MC --> PM
    SS --> RM

    BA --> BT_ENGINE
    BT_ENGINE --> BT_DATA
    BT_ENGINE --> BT_EXEC
    BT_EXEC --> BT_STATS

    LA --> LT_SYSTEM
    LT_SYSTEM --> LT_STREAM
    LT_SYSTEM --> LT_ORDERS
    LT_ORDERS --> LT_POS
    LT_POS --> LT_BROKER
```

## 🔄 Execution Flow Diagrams

### **1. Strategy Execution Flow - Backtesting Mode**

```mermaid
sequenceDiagram
    participant User as User
    participant CLI as CLI Interface
    participant UTE as Unified Engine
    participant BA as Backtest Adapter
    participant BS as Backtester System
    participant ST as Strategy

    User->>CLI: python main.py --config backtest_config.yaml
    CLI->>UTE: initialize(config)

    Note over UTE: Configuration Phase
    UTE->>UTE: load_configuration()
    UTE->>BA: initialize_backtester()
    BA->>BS: setup_historical_data()
    BS->>BS: load_csv_files()

    Note over UTE: Strategy Loading
    UTE->>ST: get_requirements()
    ST->>UTE: StrategyRequirements(timeframes, warmup)
    UTE->>BA: validate_requirements()
    BA->>UTE: requirements_satisfied

    Note over UTE: Execution Loop
    loop For each timestamp in historical data
        BA->>BS: get_market_data(timestamp)
        BS->>BA: raw_ohlcv_data
        BA->>UTE: MarketContext(data)

        UTE->>BA: get_strategy_state()
        BA->>BA: read_memory_state()
        BA->>UTE: StrategyState(variables)

        UTE->>ST: generate_signal(context, state)
        ST->>UTE: Signal (or None)

        alt Signal Generated
            UTE->>BA: execute_signal(signal)
            BA->>BS: simulate_trade(signal)
            BS->>BA: execution_result
            BA->>UTE: trade_executed

            UTE->>BA: update_state(result)
            BA->>BA: update_memory_state()
        end
    end

    BA->>BS: calculate_performance()
    BS->>BA: performance_metrics
    BA->>UTE: backtest_results
    UTE->>CLI: ExecutionResults
    CLI->>User: Display Results
```

### **2. Strategy Execution Flow - Live Trading Mode**

```mermaid
sequenceDiagram
    participant User as User
    participant CLI as CLI Interface
    participant UTE as Unified Engine
    participant LA as Live Adapter
    participant LS as Live System
    participant ST as Strategy
    participant BRK as Broker

    User->>CLI: python main.py --config live_config.yaml
    CLI->>UTE: initialize(config)

    Note over UTE: Initialization Phase
    UTE->>UTE: load_configuration()
    UTE->>LA: initialize_live_system()
    LA->>LS: setup_broker_connection()
    LS->>BRK: authenticate()
    BRK->>LS: connection_established

    Note over UTE: Strategy Setup
    UTE->>ST: get_requirements()
    ST->>UTE: StrategyRequirements
    UTE->>LA: start_data_streams()
    LA->>LS: subscribe_to_symbols()
    LS->>BRK: start_websocket_feed()

    Note over UTE: Continuous Execution Loop
    loop While market is open
        BRK->>LS: market_data_update
        LS->>LA: process_market_update()
        LA->>UTE: MarketContext(live_data)

        UTE->>LA: get_strategy_state()
        LA->>LS: read_position_file()
        LS->>LA: current_positions
        LA->>UTE: StrategyState(external_state)

        UTE->>ST: generate_signal(context, state)
        ST->>UTE: Signal (or None)

        alt Signal Generated
            UTE->>LA: execute_signal(signal)
            LA->>LS: place_order(signal)
            LS->>BRK: send_order_request()
            BRK->>LS: order_confirmation
            LS->>LA: execution_result
            LA->>UTE: trade_executed

            UTE->>LA: update_state(result)
            LA->>LS: persist_position_state()
            LS->>LS: update_position_files()
        end

        LA->>LA: wait_for_next_update()
    end

    LA->>LS: stop_data_streams()
    LS->>BRK: close_connection()
    LA->>UTE: live_session_results
    UTE->>CLI: ExecutionResults
    CLI->>User: Display Results
```

## 📈 Data Flow Architecture

### **Market Data Flow - Unified Abstraction**

```mermaid
graph LR
    subgraph "Data Sources"
        CSV[CSV Files<br/>Historical]
        WS[WebSocket<br/>Real-time]
        API[REST API<br/>Historical]
    end

    subgraph "Data Providers"
        HDP[Historical Data Provider]
        LDP[Live Data Provider]
    end

    subgraph "Market Context"
        MC[Market Context]
        TF5[5min Timeframe]
        TF15[15min Timeframe]
        CURR[Current Price]
    end

    subgraph "Strategy Layer"
        ST[Strategy Logic]
        IND[Indicator Calculations]
        SIG[Signal Generation]
    end

    CSV --> HDP
    WS --> LDP
    API --> HDP

    HDP --> MC
    LDP --> MC

    MC --> TF5
    MC --> TF15
    MC --> CURR

    TF5 --> ST
    TF15 --> ST
    CURR --> ST

    ST --> IND
    IND --> SIG
```

### **State Management Flow**

```mermaid
graph TB
    subgraph "Strategy Layer"
        ST[Strategy Logic]
        VARS[Strategy Variables]
    end

    subgraph "State Abstraction"
        SS[Strategy State Interface]
        GET[get(key)]
        SET[set(key, value)]
        POS[get_position_info()]
    end

    subgraph "Backtester State"
        MEM[In-Memory Storage]
        DICT[Python Dictionary]
        OBJ[Position Object]
    end

    subgraph "Live Trading State"
        EXT[External Storage]
        JSON[JSON Files]
        PM[Position Manager]
    end

    ST --> VARS
    VARS --> SS
    SS --> GET
    SS --> SET
    SS --> POS

    GET --> MEM
    SET --> MEM
    POS --> MEM

    GET --> EXT
    SET --> EXT
    POS --> EXT

    MEM --> DICT
    MEM --> OBJ

    EXT --> JSON
    EXT --> PM
```

## 🏗️ Repository Integration Flow

### **Current State vs Target State**

```mermaid
graph TB
    subgraph "CURRENT STATE"
        subgraph "Backtester Repo"
            BT_STRAT[MSE Backtesting Strategy]
            BT_ENGINE[Backtesting Engine]
            BT_DATA[CSV Data Handler]
        end

        subgraph "Live Module Repo"
            LT_STRAT[MSE Live Strategy]
            LT_SYSTEM[Trading System]
            LT_BROKER[Broker Integration]
        end

        BT_STRAT -.->|Duplicate Logic| LT_STRAT
    end

    subgraph "TARGET STATE"
        subgraph "Unified Repository"
            UNIFIED_STRAT[Unified MSE Strategy]
            UNIFIED_ENGINE[Unified Trading Engine]
        end

        subgraph "Backtester Adapter"
            BT_ADAPTER[Backtester Adapter]
        end

        subgraph "Live Trading Adapter"
            LT_ADAPTER[Live Trading Adapter]
        end

        UNIFIED_STRAT --> UNIFIED_ENGINE
        UNIFIED_ENGINE --> BT_ADAPTER
        UNIFIED_ENGINE --> LT_ADAPTER

        BT_ADAPTER --> BT_ENGINE
        BT_ADAPTER --> BT_DATA

        LT_ADAPTER --> LT_SYSTEM
        LT_ADAPTER --> LT_BROKER
    end
```

### **Repository Structure Flow**

```mermaid
graph LR
    subgraph "New Unified Repository"
        UR[unified_trading/]

        subgraph "Core Components"
            CORE[core/]
            STRAT[strategies/]
            ADAPT[adapters/]
            CONFIG[config/]
        end

        subgraph "External Dependencies"
            BT_MOD[backtester_module/]
            LT_MOD[live_trading_module/]
        end
    end

    subgraph "Existing Repositories"
        BT_REPO[D:/Trading/backtester/]
        LT_REPO[D:/Trading/live_module_14-04/]
    end

    UR --> CORE
    UR --> STRAT
    UR --> ADAPT
    UR --> CONFIG

    ADAPT --> BT_MOD
    ADAPT --> LT_MOD

    BT_MOD -.->|Git Submodule| BT_REPO
    LT_MOD -.->|Git Submodule| LT_REPO
```

## 🔧 MSE Strategy Migration Flow

### **Current MSE vs Unified MSE**

```mermaid
graph TB
    subgraph "CURRENT IMPLEMENTATIONS"
        subgraph "Backtester MSE"
            BT_MSE[MSEStrategyBacktesting]
            BT_EXEC[execute_strategy()]
            BT_POS[Internal position tracking]
            BT_EXIT[Exit logic in execution loop]
        end

        subgraph "Live MSE"
            LT_MSE[MSEStrategy]
            LT_SIG[generate_signal()]
            LT_EXT[External position management]
            LT_MODE[Entry/Exit mode architecture]
        end
    end

    subgraph "UNIFIED IMPLEMENTATION"
        subgraph "Unified MSE Strategy"
            UN_MSE[UnifiedMSEStrategy]
            UN_REQ[get_requirements()]
            UN_SIG[generate_signal()]
            UN_INIT[initialize_state()]
        end

        subgraph "Shared Logic"
            CALC[calculate_indicators()]
            ENTRY[check_entry_conditions()]
            EXIT[check_exit_conditions()]
            CONF[calculate_confidence()]
        end
    end

    BT_MSE -.->|Extract Logic| UN_MSE
    LT_MSE -.->|Extract Logic| UN_MSE

    UN_MSE --> UN_REQ
    UN_MSE --> UN_SIG
    UN_MSE --> UN_INIT

    UN_SIG --> CALC
    UN_SIG --> ENTRY
    UN_SIG --> EXIT
    UN_SIG --> CONF
```

## 🧪 Testing and Validation Flow

### **Signal Parity Validation Pipeline**

```mermaid
sequenceDiagram
    participant TEST as Test Framework
    participant UTE as Unified Engine
    participant BA as Backtest Adapter
    participant LA as Live Adapter
    participant VAL as Validator

    Note over TEST,VAL: Setup Phase
    TEST->>UTE: initialize_both_modes()
    UTE->>BA: setup_backtester()
    UTE->>LA: setup_live_system(mock_mode)

    Note over TEST,VAL: Data Preparation
    TEST->>TEST: load_historical_dataset()
    TEST->>BA: feed_historical_data()
    TEST->>LA: simulate_live_feed()

    Note over TEST,VAL: Signal Generation
    loop For each timestamp
        TEST->>BA: generate_signal_backtest()
        BA->>TEST: backtest_signal

        TEST->>LA: generate_signal_live()
        LA->>TEST: live_signal

        TEST->>VAL: compare_signals(bt_signal, live_signal)
        VAL->>TEST: comparison_result

        alt Signals don't match
            TEST->>TEST: log_divergence()
            TEST->>VAL: analyze_root_cause()
        end
    end

    TEST->>VAL: generate_validation_report()
    VAL->>TEST: validation_results
```

### **Performance Validation Flow**

```mermaid
graph LR
    subgraph "Input Data"
        HIST[Historical Data]
        CONFIG[Test Configuration]
    end

    subgraph "Execution Engines"
        BT_ENG[Backtest Engine]
        LIVE_ENG[Live Engine (Paper)]
    end

    subgraph "Signal Capture"
        BT_SIG[Backtest Signals]
        LIVE_SIG[Live Signals]
    end

    subgraph "Validation Layer"
        COMP[Signal Comparator]
        STATS[Statistical Analysis]
        REPORT[Validation Report]
    end

    HIST --> BT_ENG
    HIST --> LIVE_ENG
    CONFIG --> BT_ENG
    CONFIG --> LIVE_ENG

    BT_ENG --> BT_SIG
    LIVE_ENG --> LIVE_SIG

    BT_SIG --> COMP
    LIVE_SIG --> COMP

    COMP --> STATS
    STATS --> REPORT
```

## 🚀 Deployment Flow

### **Staged Deployment Pipeline**

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_CODE[Unified Codebase]
        DEV_TEST[Unit Tests]
        DEV_INT[Integration Tests]
    end

    subgraph "Staging Environment"
        STAGE_DEPLOY[Staging Deployment]
        STAGE_VAL[Signal Validation]
        STAGE_PERF[Performance Testing]
    end

    subgraph "Production Environment"
        PROD_SHADOW[Shadow Trading]
        PROD_LIMITED[Limited Live Trading]
        PROD_FULL[Full Production]
    end

    DEV_CODE --> DEV_TEST
    DEV_TEST --> DEV_INT
    DEV_INT --> STAGE_DEPLOY

    STAGE_DEPLOY --> STAGE_VAL
    STAGE_VAL --> STAGE_PERF
    STAGE_PERF --> PROD_SHADOW

    PROD_SHADOW --> PROD_LIMITED
    PROD_LIMITED --> PROD_FULL
```

### **Rollback Strategy Flow**

```mermaid
graph LR
    subgraph "Monitoring"
        ALERT[Performance Alert]
        THRESH[Threshold Breach]
    end

    subgraph "Decision"
        EVAL[Evaluate Severity]
        DECIDE[Rollback Decision]
    end

    subgraph "Rollback Actions"
        STOP[Stop Unified System]
        REVERT[Revert to Legacy]
        VERIFY[Verify Legacy Function]
    end

    ALERT --> EVAL
    THRESH --> EVAL
    EVAL --> DECIDE
    DECIDE --> STOP
    STOP --> REVERT
    REVERT --> VERIFY
```

---

## 🎯 Flow Summary

These diagrams illustrate:

1. **System Integration**: How unified engine coordinates with existing systems
2. **Data Flow**: How market data flows through abstraction layers
3. **State Management**: How strategy state is handled differently across environments
4. **Repository Structure**: How to organize code for maximum reusability
5. **Migration Strategy**: How to move from current to unified implementation
6. **Validation Process**: How to ensure identical behavior across environments
7. **Deployment Pipeline**: How to safely roll out the unified system

Each flow diagram can be implemented incrementally, allowing for staged development and validation of the unified trading system.

**Next**: Review [Implementation Plan](../implementation/IMPLEMENTATION_PLAN.md) for detailed development steps.